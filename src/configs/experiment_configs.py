from src.configs.configs import (
    OptimizerConfig,
    RMSPROPConfig,
    SGDConfig,
    ADAMConfig,
    NLCGConfig,
    TaskType,
    TrainConfig,
    DataConfig,
)
from src.utils import custom

optimizers: dict[str, OptimizerConfig] = {
    oc.name: oc
    for oc in [
        RMSPROPConfig(name="RMSPROP", learning_rate=1e-3, rho=0.9, epsilon=1e-7),
        SGDConfig(name="SGD", learning_rate=0.01, momentum=0.0),
        ADAMConfig(
            name="ADAM", learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7
        ),
        NLCGConfig(
            name="NLCG",
            model=None,
            loss=None,
            max_iters=3,
            tol=1e-7,
            c1=1e-4,
            c2=0.1,
            amax=1.0,
        ),
    ]
}

train: dict[TaskType, TrainConfig] = {
    TaskType.REGRESSION: TrainConfig(
        seed=42,
        max_calls=12_000,
        max_epochs=1_000,
        loss_fn=custom.mse,
        batch_size=None,
        metrics=[],
    ),
    TaskType.BINARY_CLASSIFICATION: TrainConfig(
        seed=42,
        max_calls=12_000,
        max_epochs=1_000,
        loss_fn=custom.bin_crossentropy,
        batch_size=None,
        metrics=[custom.bin_accuracy],
    ),
    TaskType.MULTICLASS_CLASSIFICATION: TrainConfig(
        seed=42,
        max_calls=36000,
        max_epochs=10,
        loss_fn=custom.crossentropy,
        batch_size=None,
        metrics=[custom.accuracy],
    ),
}

data: dict[str, DataConfig] = {
    "MNIST": DataConfig(
        path=None,  # set in CLI
        name="MNIST",
        task=TaskType.MULTICLASS_CLASSIFICATION,
        num_classes=10,
        input_shape=(1, 28, 28, 1),
    ),
    "IMAGENET": DataConfig(
        path=None,  # set in CLI
        name="IMAGENET",
        task=TaskType.MULTICLASS_CLASSIFICATION,
        num_classes=1000,
        input_shape=(1, 224, 224, 3),
    ),
    "CIFAR10": DataConfig(
        path=None,
        name="CIFAR10",
        task=TaskType.MULTICLASS_CLASSIFICATION,
        num_classes=10,
        input_shape=(1, 32, 32, 3),
    ),
    "CIFAR100": DataConfig(
        path=None,
        name="CIFAR100",
        task=TaskType.MULTICLASS_CLASSIFICATION,
        num_classes=100,
        input_shape=(1, 32, 32, 3),
    ),
}

dtypes: frozenset = frozenset(("float32", "float64"))
