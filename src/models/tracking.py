import dataclasses
from collections import defaultdict
from typing import Optional, Callable
import numpy as np
from tqdm import tqdm
import tensorflow as tf

__all__ = ["CustomTqdmCallback", "TrainTrack", "compute_full_loss"]


class CustomTqdmCallback(tqdm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_postfix = {}

    def update_to(self, amount: int = None, **postfix):
        if amount is not None:
            self.update(amount - self.n)
        self.cached_postfix.update(postfix)
        self.set_postfix(**self.cached_postfix)


@dataclasses.dataclass(slots=True)
class TrainTrack:
    # Number of times the function/model is called
    nb_function_calls: int = 0
    # Number of times the gradient is called
    nb_gradient_calls: int = 0
    # The current loss value over the full data
    loss: float = np.inf
    val_loss: float | None = None
    # The current epoch: One epoch is one go through the full dataset
    epoch: int = 0
    # Tracking the best epoch/iteration/loss (note: this is the train loss)
    best_epoch: int = 0
    best_loss: float = np.inf
    best_val_epoch: int | None = None
    best_val_loss: float = np.inf
    metrics: dict[str, float] = dataclasses.field(
        default_factory=lambda: defaultdict(lambda: -np.inf)
    )

    @property
    def steps(self):
        return self.nb_function_calls + self.nb_gradient_calls

    def to_dict(self, target: Optional[str] = None) -> dict:
        for_summary = dict(
            best_epoch=self.best_epoch,
            best_val_epoch=self.best_val_epoch,
            best_loss=self.best_loss,
            best_val_loss=self.best_val_loss,
        )
        if target is not None and target == "summary":
            return for_summary

        for_log = dict(
            nb_function_calls=self.nb_function_calls,
            nb_gradient_calls=self.nb_gradient_calls,
            steps=self.steps,
            epoch=self.epoch,
            loss=self.loss,
            val_loss=self.val_loss,
            **self.metrics,
        )
        if target is not None and target == "log":
            return for_log

        if target is None:
            return for_log | for_summary

        raise ValueError(f"Unknown target: {target}")
    
def compute_full_loss(
    model: tf.keras.Model, loss_fn: Callable, data: tf.data.Dataset
) -> float | None:
    initial_loss = tf.keras.metrics.Mean()

    for x, y in data:
        initial_loss.update_state(loss_fn(y_true=y, y_pred=model(x)))

    return float(initial_loss.result())
