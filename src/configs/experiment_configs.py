from src.configs.configs import (
    OptimizerConfig,
    RMSPROPConfig,
    SGDConfig,
    ADAMConfig,
    NLCGConfigEager,
    NLCGConfig,
    TaskType,
    TrainConfig,
    DataConfig,
    ModelConfig,
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
        NLCGConfigEager(
            name="NLCGEager",
            model=None,
            loss=None,
            max_iters=50,
            tol=1e-7,
            c1=1e-4,
            c2=0.9,
            amax=1.0,
        ),
        NLCGConfig(
            name="NLCG",
            model=None,
            loss=None,
            max_iters=50,
            tol=1e-7,
            c1=1e-4,
            c2=0.9,
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
        max_calls=2500000,
        max_epochs=20,
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
        num_units_mlp=76,
        num_base_filters=300,
        input_shape=(1, 28, 28, 1),
    ),
    "FASHION_MNIST": DataConfig(
        path=None,  # set in CLI
        name="FASHION_MNIST",
        task=TaskType.MULTICLASS_CLASSIFICATION,
        num_classes=10,
        num_units_mlp=76,
        num_base_filters=300,
        input_shape=(1, 28, 28, 1),
    ),
    "IMAGENET": DataConfig(
        path=None,  # set in CLI
        name="IMAGENET",
        task=TaskType.MULTICLASS_CLASSIFICATION,
        num_classes=1000,
        num_units_mlp=10,
        num_base_filters=None,
        input_shape=(1, 224, 224, 3),
    ),
    "CIFAR10": DataConfig(
        path=None,
        name="CIFAR10",
        task=TaskType.MULTICLASS_CLASSIFICATION,
        num_classes=10,
        num_units_mlp=16,
        num_base_filters=280,
        input_shape=(1, 32, 32, 3),
    ),
    "CIFAR100": DataConfig(
        path=None,
        name="CIFAR100",
        task=TaskType.MULTICLASS_CLASSIFICATION,
        num_classes=100,
        num_units_mlp=160,
        num_base_filters=260,
        input_shape=(1, 32, 32, 3),
    ),
}

models: dict[str, ModelConfig] = {
    "FlatMLP": ModelConfig(name="FlatMLP"),
    "FlatCNN": ModelConfig(name="FlatCNN"),
    "FlatCNNCifar100": ModelConfig(name="FlatCNNCifar100"),
    "BasicCNN": ModelConfig(name="BasicCNN"),
    "CIFARCNN": ModelConfig(name="CIFARCNN"),
    "ResNetTypeI_18": ModelConfig(name="ResNetTypeI_18"),
    "ResNetTypeI_34": ModelConfig(name="ResNetTypeI_34"),
    "ResNetTypeII_50": ModelConfig(name="ResNetTypeII_50"),
    "ResNetTypeII_101": ModelConfig(name="ResNetTypeII_101"),
    "ResNetTypeII_152": ModelConfig(name="ResNetTypeII_152"),
    "ResNetCIFAR": ModelConfig(name="ResNetCIFAR"),
}

dtypes: frozenset = frozenset(("float32", "float64"))

gpus: frozenset = frozenset(
    (
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
        "11",
        "12",
        "13",
        "14",
        "15",
    )
)
