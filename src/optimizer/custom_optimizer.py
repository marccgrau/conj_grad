import tensorflow as tf
import keras.optimizers.experimental as optimizers
from keras.optimizers.optimizer_v2 import utils as optimizer_utils
from tensorflow_probability.python.optimizer.linesearch.hager_zhang import hager_zhang

from src.utils.custom import as_Kfloat

keras = tf.keras
K = keras.backend

class CustomOptimizer(optimizers.Optimizer):
    def __init__(self, c1, c2, tol, max_iter, **kwargs):
        super().__init__(c1, c2, tol, max_iter, **kwargs)
        self.c1 = c1
        self.c2 = c2
        self.tol = tol
        self.max_iter = max_iter
    
    def build(self, var_list):
        super().build(var_list)
        raise NotImplementedError
    
    def update_step():
        raise NotImplementedError
    
    def get_config():
        raise NotImplementedError
    
    def apply_gradients(self, grads_and_vars, name=None):
        """Apply gradients to variables.
        Args:
          grads_and_vars: List of `(gradient, variable)` pairs.
          name: string, defaults to None. The name of the namescope to
            use when creating variables. If None, `self.name` will be used.
        Returns:
          A `tf.Variable`, representing the current iteration.
        Raises:
          TypeError: If `grads_and_vars` is malformed.
        """
        # TODO: should be adapted according to line search
        self._compute_current_learning_rate() 
        grads_and_vars = list(grads_and_vars)
        if len(grads_and_vars) == 0:
            # It is possible that the grad is empty. In this case,
            # `apply_gradients` is a no-op.
            return self._iterations
        grads, trainable_variables = zip(*grads_and_vars)
        scope_name = name or self.name or "optimizer"
        with tf.name_scope(scope_name):
            with tf.init_scope():
                # Lift variable creation to init scope to avoid environment
                # issues.
                self.build(trainable_variables)
        grads_and_vars = list(zip(grads, trainable_variables))
        grads_and_vars = optimizer_utils.filter_empty_gradients(grads_and_vars)
        if len(list(grads_and_vars)) == 0:
            # Check again after filtering gradients.
            return self._iterations

        grads, trainable_variables = zip(*grads_and_vars)

        grads = self._clip_gradients(grads)
        grads = self._deduplicate_sparse_grad(grads)
        self._apply_weight_decay(trainable_variables)
        grads_and_vars = list(zip(grads, trainable_variables))
        iteration = self._internal_apply_gradients(grads_and_vars)

        # Apply variable constraints after applying gradients.
        for variable in trainable_variables:
            if variable.constraint is not None:
                variable.assign(variable.constraint(variable))
        return iteration
    
    
    
    