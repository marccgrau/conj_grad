import tensorflow as tf

from src.configs.configs import (
    OptimizerConfig,
    ADAMConfig,
    RMSPROPConfig,
    SGDConfig,
    NLCGConfigEager,
    NLCGConfig,
)
from src.utils.custom import as_Kfloat
from src.optimizer.cg_optimizer_eager import NonlinearCGEager
from src.optimizer.cg_optimizer import NonlinearCG
from src.utils import setup


def fetch_optimizer(optimizer_config: OptimizerConfig, model, loss):
    if isinstance(optimizer_config, RMSPROPConfig):
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate=as_Kfloat(optimizer_config.learning_rate),
            rho=as_Kfloat(optimizer_config.rho),
            epsilon=as_Kfloat(optimizer_config.epsilon),
        )
    elif isinstance(optimizer_config, SGDConfig):
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=as_Kfloat(optimizer_config.learning_rate),
            momentum=as_Kfloat(optimizer_config.momentum),
        )
    elif isinstance(optimizer_config, ADAMConfig):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=as_Kfloat(optimizer_config.learning_rate),
            beta_1=as_Kfloat(optimizer_config.beta_1),
            beta_2=as_Kfloat(optimizer_config.beta_2),
            epsilon=as_Kfloat(optimizer_config.epsilon),
        )
    elif isinstance(optimizer_config, NLCGConfigEager):
        optimizer = NonlinearCGEager(
            model=model,
            loss=loss,
            max_iters=optimizer_config.max_iters,
            tol=as_Kfloat(optimizer_config.tol),
            c1=as_Kfloat(optimizer_config.c1),
            c2=as_Kfloat(optimizer_config.c2),
            amax=as_Kfloat(optimizer_config.amax),
        )
    elif isinstance(optimizer_config, NLCGConfig):
        optimizer = NonlinearCG(
            model=model,
            loss=loss,
            max_iters=optimizer_config.max_iters,
            tol=as_Kfloat(optimizer_config.tol),
            c1=as_Kfloat(optimizer_config.c1),
            c2=as_Kfloat(optimizer_config.c2),
            amax=as_Kfloat(optimizer_config.amax),
        )
    else:
        raise ValueError("Optimizer not defined.")

    return optimizer
