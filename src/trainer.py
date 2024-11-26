"""A generic training wrapper."""
import time
from copy import deepcopy
from statistics import mean
from typing import Callable, List, Union

import torch
from alive_progress import alive_bar
from numpy import mean
from torch import nn
from torch.utils.data import DataLoader

from configuration.train_config import _TrainerConfig
from df_logger import main_logger
from src.utils import save_model
from utils.utils import timeit


class Trainer:
    def __init__(
            self,
            epochs: int = 20,
            batch_size: int = 32,
            device: str = "cpu",
            optimizer_fn: Callable = torch.optim.Adam,
            optimizer_kwargs: dict = {"lr": 1e-3},
            use_scheduler: bool = False,
    ) -> None:
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.optimizer_fn = optimizer_fn
        self.optimizer_kwargs = optimizer_kwargs
        self.epoch_test_losses: List[float] = []
        self.use_scheduler = use_scheduler


def forward_and_loss(model, criterion, batch_x, batch_y, **kwargs):
    batch_out = model(batch_x)
    batch_loss = criterion(batch_out, batch_y)
    return batch_out, batch_loss


class GDTrainer(Trainer):
    def __init__(
        self,
        config: _TrainerConfig,
        device: Union[torch.device, str],
        use_scheduler: bool = False,
        checkpoint_path: str = "checkpoints"
    ):
        super().__init__(config.num_epochs, config.batch_size, device, config.optimizer, config.optimizer_parameters,
                         use_scheduler)
        self.epochs = config.num_epochs
        self.device = device
        self.optimizer_fn = config.optimizer
        self.optimizer_kwargs = config.optimizer_parameters
        self.use_scheduler = use_scheduler
        self.checkpoint_path = checkpoint_path
        self.early_stopping = config.early_stopping
        self.early_stopping_patience = config.early_stopping_patience

    @timeit("Train", "model")
    def train(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        model: nn.Module
    ):
        optimizer = self.optimizer_fn(model.parameters(), **self.optimizer_kwargs)

        # Calculate class weights based on the dataframe
        value_counts = train_loader.dataset.samples_df['label'].value_counts()
        class_counts = [value_counts['spoof'], value_counts['bonafide']]
        class_counts = torch.tensor(class_counts, dtype=torch.float32)

        class_weights = 1.0 / class_counts
        class_weights /= class_weights.sum()
        class_weights = class_weights.to(self.device)

        # Define criterion with class weights
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
        scheduler = self._create_scheduler(optimizer, train_loader) if self.use_scheduler else None

        best_model = None
        best_acc = 0
        early_stopping_counter = 0
        times = []
        main_logger.info(f"Starting training for {self.epochs} epochs, lr_opt={self.optimizer_kwargs}!")

        with alive_bar(self.epochs, bar='smooth', spinner='dots_waves2', title='Training:', force_tty=True) as bar:
            for epoch in range(self.epochs):
                bar()
                main_logger.info(f"\tEpoch num: {epoch}")

                start_time = time.time()
                train_loss, train_accuracy = self.train_epoch(model, train_loader, criterion, optimizer, scheduler,
                                                              epoch)
                epoch_time = time.time() - start_time
                times.append(epoch_time)

                main_logger.info(f"Saving model for epoch {epoch}")
                # model
                save_model(model, self.checkpoint_path, f"epoch_{epoch}.pth")

                test_loss, test_acc = self.validate(model, test_loader, criterion)

                self._log_epoch_results(epoch, train_loss, train_accuracy, test_loss, test_acc, epoch_time)

                best_model, best_acc, early_stopping_counter = self._update_best_model(
                    model, test_acc, best_acc, best_model, early_stopping_counter, epoch
                )

                if self.early_stopping and self._should_stop_early(early_stopping_counter, self.early_stopping_patience,
                                                                   test_acc):
                    break

        main_logger.info(f"Model {type(model)} Mean time: {mean(times)}")
        model.load_state_dict(best_model, strict=False)
        return model

    @timeit("Train epoch")
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        epoch: int
    ):
        model.train()
        running_loss = 0
        num_correct = 0
        num_total = 0

        for i, (batch_x, batch_y) in enumerate(train_loader):
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

            batch_out, batch_loss = self._forward_and_loss(model, criterion, batch_x, batch_y)
            batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
            num_correct += (batch_pred == batch_y.int()).sum(dim=0).item()
            running_loss += batch_loss.item() * batch_size

            if i % 100 == 0:
                self._log_batch_progress(i, epoch, running_loss, num_correct, num_total)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        return running_loss / num_total, (num_correct / num_total) * 100

    @timeit("Validation")
    def validate(self, model, test_loader, criterion):
        model.eval()
        running_loss = 0
        num_correct = 0
        num_total = 0

        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_size = batch_x.size(0)
                num_total += batch_size
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.unsqueeze(1).type(torch.float32).to(self.device)

                batch_pred = model(batch_x)
                batch_loss = criterion(batch_pred, batch_y)

                running_loss += batch_loss.item() * batch_size

                batch_pred = torch.sigmoid(batch_pred)
                batch_pred_label = (batch_pred + 0.5).int()
                num_correct += (batch_pred_label == batch_y.int()).sum(dim=0).item()

        num_total = max(num_total, 1)  # Avoid division by zero
        return running_loss / num_total, 100 * (num_correct / num_total)

    @staticmethod
    def _create_scheduler(optimizer, train_loader):
        batches_per_epoch = len(train_loader) * 2  # every 2nd epoch
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=batches_per_epoch,
            T_mult=1,
            eta_min=5e-6,
        )

    @staticmethod
    def _forward_and_loss(model, criterion, batch_x, batch_y):
        batch_out = model(batch_x)
        return batch_out, criterion(batch_out, batch_y)

    @staticmethod
    def _log_batch_progress(i, epoch, running_loss, num_correct, num_total):
        main_logger.info(
            f"\t[{epoch:04d}][{i:05d}]: {running_loss / num_total} {num_correct / num_total * 100}"
        )

    def _log_epoch_results(self, epoch, train_loss, train_accuracy, test_loss, test_acc, epoch_time):
        main_logger.info(
            f"Epoch [{epoch + 1}/{self.epochs}]: "
            f"train/loss: {train_loss}, train/accuracy: {train_accuracy}, "
            f"test/loss: {test_loss}, test/accuracy: {test_acc}, "
            f"Time = {epoch_time}"
        )

    def _update_best_model(self, model, test_acc, best_acc, best_model, early_stopping_counter, epoch_num):
        if best_model is None or test_acc > best_acc:
            best_acc = test_acc
            best_model = deepcopy(model.state_dict())
            early_stopping_counter = 0
            save_model(model, self.checkpoint_path, f"epoch{epoch_num}_{best_acc:.3f}.pth")
            main_logger.info(f'New best model with accuracy: {best_acc:.3f}')
        elif self.early_stopping:
            early_stopping_counter += 1
            main_logger.info(f'No improvement. Early stopping counter: {early_stopping_counter}/5')
        return best_model, best_acc, early_stopping_counter

    def _should_stop_early(self, early_stopping_counter, patience, test_acc):
        if self.early_stopping and early_stopping_counter >= patience or round(test_acc, 3) == 100.000:
            main_logger.info('Early stopping triggered.')
            return True
        return False

    @staticmethod
    def _compare_metrics(metric_1: float, metric_2: float, maximise: bool = True) -> bool:
        """
        Compares two metrics based on the specified condition.
        """

        return metric_1 > metric_2 if maximise else metric_1 < metric_2
