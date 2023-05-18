import dataclasses
import enum
from copy import deepcopy
from pathlib import Path
from typing import Optional, Callable
import tensorflow as tf


class TaskType(enum.Enum):
    REGRESSION = 1
    BINARY_CLASSIFICATION = 2
    MULTICLASS_CLASSIFICATION = 3


@dataclasses.dataclass(slots=True)
class DataConfig:
    name: str
    task: TaskType
    path: Optional[Path] = dataclasses.field(default=None)
    num_classes: Optional[int] = dataclasses.field(default=None)
    input_shape: tuple[int, int, int, int] = dataclasses.field(default=None)

    @property
    def full_name(self):
        return f"{self.name}"


@dataclasses.dataclass(slots=True)
class ModelConfig:
    name: str


@dataclasses.dataclass(slots=True)
class TrainConfig:
    seed: int  # Seed for replication
    max_calls: int  # Maximum number of function call equivalents
    max_epochs: int  # Maximum number of epoch (iterations of the full dataset)
    loss_fn: Optional[Callable[[tf.Tensor, tf.Tensor], tf.Tensor]]  # can be dynamic
    batch_size: None | int
    metrics: list[Callable]

    def __repr__(self):
        fn = self.loss_fn.__name__ if self.loss_fn is not None else "None"
        return f"TrainConfig(max_calls='{self.max_calls}', max_epochs='{self.max_epochs}', loss_fn={fn}, batch_size={self.batch_size})"


_KERAS_OPTIMIZERS = frozenset({"RMSPROP", "SGD", "ADAM", "NLCG"})


@dataclasses.dataclass(slots=True, frozen=True)
class OptimizerConfig:
    name: str

    @property
    def is_keras(self) -> bool:
        return self.name in _KERAS_OPTIMIZERS


@dataclasses.dataclass(slots=True, frozen=True)
class KerasOptimizerConfig(OptimizerConfig):
    pass


@dataclasses.dataclass(slots=True, frozen=True)
class ScipyOptimizerConfig(OptimizerConfig):
    pass


@dataclasses.dataclass(slots=True, frozen=True)
class RMSPROPConfig(KerasOptimizerConfig):
    learning_rate: float
    rho: float
    epsilon: float


@dataclasses.dataclass(slots=True, frozen=True)
class SGDConfig(KerasOptimizerConfig):
    learning_rate: float
    momentum: float


@dataclasses.dataclass(slots=True, frozen=True)
class ADAMConfig(KerasOptimizerConfig):
    learning_rate: float
    beta_1: float
    beta_2: float
    epsilon: float


@dataclasses.dataclass(slots=True, frozen=True)
class NLCGConfigEager(KerasOptimizerConfig):
    model: tf.keras.Model
    loss: tf.keras.losses.Loss
    max_iters: int
    tol: float
    c1: float
    c2: float
    amax: float


@dataclasses.dataclass(slots=True, frozen=True)
class NLCGConfig(KerasOptimizerConfig):
    model: tf.keras.Model
    loss: tf.keras.losses.Loss
    max_iters: int
    tol: float
    c1: float
    c2: float
    amax: float
