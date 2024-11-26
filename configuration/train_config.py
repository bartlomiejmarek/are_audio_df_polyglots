from os import makedirs
from typing import Optional, Type

from pydantic import Field, BaseModel
from torch.nn import Module, BCEWithLogitsLoss
from torch.optim import Adam, Optimizer
from torch.optim.lr_scheduler import LRScheduler


class _TrainerConfig(BaseModel):
    num_epochs: int = Field(default=25, description="Number of epochs to train the model")
    batch_size: int = Field(default=32, description="Batch size for training")
    validate_every: int = Field(default=1, description="Number of epochs after which to validate the model")
    checkpoint_dir: str = Field(default="checkpoints", description="Directory to save model checkpoints")
    optimizer_parameters: dict = Field(default={}, description="Parameters for the optimizer")
    optimizer: Optional[Type[Optimizer]] = Field(default=Adam, description="Parameters for the optimizer")
    scheduler: Optional[Type[LRScheduler]] = Field(default=None, description="Scheduler for the optimizer")
    scheduler_parameters: dict = Field(default={}, description="Parameters for the scheduler")
    early_stopping: bool = Field(default=True, description="Whether to use early stopping")
    early_stopping_patience: int = Field(default=5, description="Number of epochs to wait before stopping training")

    def __init__(self, **data):
        super().__init__(**data)
        makedirs(self.checkpoint_dir, exist_ok=True)

    class Config:
        """Pydantic configuration"""
        extra = "allow"  # Allows child classes to add extra fields
        arbitrary_types_allowed = True


if __name__ == '__main__':
    print(_TrainerConfig().model_dump())

