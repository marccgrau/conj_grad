import tensorflow as tf

from src.configs.configs import OptimizerConfig, ADAMConfig, RMSPROPConfig, SGDConfig, NLCGConfig, CustomConfig
from src.utils.custom import as_Kfloat
from src.optimizer.cg_optimizer_eager import NonlinearCGEager
from src.optimizer.custom_optimizer import CustomOptimizer

def fetch_optimizer(optimizer_config: OptimizerConfig):
    if isinstance(optimizer_config, RMSPROPConfig):
        optimizer = tf.keras.optimizers.RMSprop(
            learning_rate = as_Kfloat(optimizer_config.learning_rate),
            rho = as_Kfloat(optimizer_config.rho),
            epsilon = as_Kfloat(optimizer_config.epsilon),
        )
    elif isinstance(optimizer_config, SGDConfig):
        optimizer = tf.keras.optimizers.SGD(
            learning_rate = as_Kfloat(optimizer_config.learning_rate),
            momentum = as_Kfloat(optimizer_config.momentum),
        )
    elif isinstance(optimizer_config, ADAMConfig):
        optimizer = tf.keras.optimizers.Adam(
            learning_rate = as_Kfloat(optimizer_config.learning_rate),
            beta_1 = as_Kfloat(optimizer_config.beta_1),
            beta_2 = as_Kfloat(optimizer_config.beta_2),
            epsilon = as_Kfloat(optimizer_config.epsilon),
        )
    elif isinstance(optimizer_config, NLCGConfig):
        optimizer = NonlinearCGEager(
            model=optimizer_config.model,
            loss=optimizer_config.loss,
            max_iters=optimizer_config.max_iters,
            tol=optimizer_config.tol,
            c1=optimizer_config.c1,
            c2=optimizer_config.c2,
            amax=optimizer_config.amax,
        )
    else:
        raise ValueError("Optimizer not defined.")
    
    return optimizer
    