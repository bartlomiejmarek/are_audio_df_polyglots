from os.path import exists
from pathlib import Path
from typing import Dict
from typing import List, Union

import numpy as np
import torch
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, accuracy_score, roc_curve
from torch.utils.data import DataLoader

from df_logger import main_logger, Logger
from src.models import get_model


def calculate_far_frr_eer(
        y_true: Union[List[int], np.ndarray],
        y_scores: Union[List[float], np.ndarray]
):
    """
    Calculate FAR and FRR for a given threshold.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr, kind='linear')(x), 0., 1.)
    threshold = interp1d(fpr, thresholds, kind='linear')(eer)
    return fpr, tpr, eer, threshold


def _calculate_metrics(all_labels, all_preds, all_pred_label) -> Dict[str, float]:
    accuracy = accuracy_score(all_labels.cpu().numpy(), all_pred_label.cpu().numpy())

    auc_score = roc_auc_score(y_true=all_labels.cpu().numpy(), y_score=all_preds.cpu().numpy())

    # For EER flip values, following original evaluation implementation

    fpr, tpr, eer, threshold = calculate_far_frr_eer(
        y_true=all_labels.cpu().numpy(),
        y_scores=all_preds.cpu().numpy(),
    )

    precision, recall, f1_score, support = precision_recall_fscore_support(
        all_labels.cpu().numpy(), all_pred_label.cpu().numpy(), average="binary", beta=1.0
    )

    return {
        "accuracy": accuracy,
        'auc': auc_score,
        'eer': eer,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'support': support,
    }


def log_metrics(logger: Logger, prefix: str, metrics: Dict[str, float]) -> None:
    logger.info(f"{prefix} metrics: " + " | ".join([f"{k}: {100 * float(v):.4f}" if isinstance(v, (
        int, float)) or (hasattr(v, 'size') and v.size == 1) else f"{k}: {v}" for k, v in metrics.items() if
                                                    v is not None]))


def evaluate_nn(
        model_path: Union[str, Path],
        test_loader: DataLoader,
        model_config: Dict,
        device: Union[torch.device, str]
):
    model_name, model_parameters = model_config["name"], model_config["parameters"]

    # Load model architecture
    model = get_model(
        model_name=model_name,
        config=model_parameters,
        device=device
    )

    # If provided weights, apply corresponding ones (from an appropriate fold)
    if model_path is not None and exists(model_path):
        model.load_state_dict(torch.load(Path(model_path)), strict=False)
    model = model.to(device)

    main_logger.info(
        f"Testing '{model_name}' model, weights path: '{model_path}', on {len(test_loader)} audio files."
    )

    y_pred = torch.Tensor([]).to(device)
    y = torch.Tensor([]).to(device)
    y_pred_label = torch.Tensor([]).to(device)

    for i, (batch_x, batch_y) in enumerate(test_loader):
        model.eval()

        with torch.no_grad():
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            batch_pred = torch.sigmoid(model(batch_x).squeeze(1))
            batch_pred_label = torch.round(batch_pred).int()

            y_pred = torch.cat([y_pred, batch_pred], dim=0)
            y_pred_label = torch.cat([y_pred_label, batch_pred_label], dim=0)
            y = torch.cat([y, batch_y], dim=0)

            del batch_x, batch_y, batch_pred
    torch.cuda.empty_cache()
    return y, y_pred, y_pred_label

