from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.framework import ops
from tensorflow.python.training import training_ops
import tensorflow as tf
from tensorflow.keras.optimizers.experimental import Optimizer

class NLCGOptimizer(Optimizer):
    """
    Would require to initially set weights and loss
    
    optimizer.set_weights({'f_prev': initial_loss, 'x_prev': initial_weights})
    
    These would serve as first inputs to the optimizer.

    """
    def __init__(self, alpha=0.01, epsilon=1e-7, use_locking=False, name="NLCGOptimizer", **kwargs):
        super(NLCGOptimizer, self).__init__(use_locking, name, **kwargs)
        self.alpha = alpha
        self.epsilon = epsilon
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'previous_grad')
    
    # armijo-goldstein line search
    def _line_search(self, f0, x0, p, alpha, beta, max_iters):
        f_prev, x_prev = f0, x0
        for i in range(max_iters):
            fx, _, _ = self._resource_apply_dense(p, x_prev)
            if fx > f_prev:
                break
            alpha *= beta
            f_prev, x_prev = fx, x_prev + alpha * p
        return f_prev, x_prev, alpha
    
    def line_search_wolfe(f, x, grad, direction, c1=1e-4, c2=0.9, max_iters=20):
        """
        Backtracking line search that satisfies the Wolfe conditions.
        
        Parameters:
            f (callable): Objective function.
            x (tf.Tensor): Current point.
            grad (tf.Tensor): Gradient of objective function at current point.
            direction (tf.Tensor): Search direction.
            c1 (float): Armijo-Goldstein parameter.
            c2 (float): Curvature parameter.
            max_iters (int): Maximum number of iterations.
            
        Returns:
            alpha (tf.Tensor): Step size.
        """
        alpha = tf.constant(1.0, dtype=x.dtype)
        fx, _, _ = f(x)
        gx = tf.reduce_sum(grad * direction)
        
        for i in range(max_iters):
            x_next = x + alpha * direction
            fx_next, gx_next, _ = f(x_next)
            if fx_next > fx + c1 * alpha * gx or (i > 0 and fx_next >= fx_next_prev):
                # Armijo-Goldstein condition not satisfied, or objective function not decreasing
                alpha_low = 0.0
                alpha_high = alpha
                for j in range(10):
                    alpha = (alpha_low + alpha_high) / 2.0
                    x_next = x + alpha * direction
                    fx_next, gx_next, _ = f(x_next)
                    if fx_next > fx + c1 * alpha * gx or fx_next >= fx_next_prev:
                        alpha_high = alpha
                    else:
                        gx_next = tf.reduce_sum(grad(x_next) * direction)
                        if gx_next <= c2 * gx:
                            return alpha
                        alpha_low = alpha
                return alpha
            gx_next = tf.reduce_sum(grad(x_next) * direction)
            if tf.abs(gx_next) <= -c2 * gx:
                return alpha
            if gx_next >= 0:
                # Curvature condition not satisfied, zoom in
                alpha_low = alpha
                alpha_high = alpha_prev
                for j in range(10):
                    alpha = (alpha_low + alpha_high) / 2.0
                    x_next = x + alpha * direction
                    fx_next, gx_next, _ = f(x_next)
                    if fx_next > fx + c1 * alpha * gx or fx_next >= fx_next_prev:
                        alpha_high = alpha
                    else:
                        gx_next = tf.reduce_sum(grad(x_next) * direction)
                        if tf.abs(gx_next) <= -c2 * gx:
                            return alpha
                        if gx_next * (alpha_high - alpha_low) >= 0:
                            alpha_high = alpha_low
                        alpha_low = alpha
                return alpha
            alpha_prev = alpha
            fx_next_prev = fx_next
        
        return alpha

    def _resource_apply_dense(self, grad, var):
        previous_grad = self.get_slot(var, 'previous_grad')
        if previous_grad is None:
            previous_grad = var.assign(tf.zeros_like(var))
        g_norm_squared = tf.reduce_sum(tf.square(grad))
        if g_norm_squared == 0:
            return tf.no_op()
        if previous_grad is None:
            previous_grad = tf.zeros_like(var)
        beta = tf.reduce_sum(grad * (grad - previous_grad)) / tf.maximum(tf.reduce_sum(tf.square(previous_grad)), self._get_hyper('epsilon'))
        direction = -grad + beta * previous_grad
        alpha = self._get_hyper('alpha')
        f_prev = self._get_hyper('f_prev')
        x_prev = self._get_hyper('x_prev')
        fx, x_next, alpha = self._line_search(f_prev, x_prev, direction, alpha, 0.1, 10)
        var_update = training_ops.resource_apply_gradient_descent(var.handle, x_next - var, use_locking=self._use_locking)
        previous_grad.assign(grad)
        self._set_hyper('f_prev', fx)
        self._set_hyper('x_prev', x_next)
        self._set_hyper('alpha', alpha)
        return var_update
    
    def get_config(self):
        config = super(NLCGOptimizer, self).get_config()
        config.update({
            'alpha': self._serialize_hyperparameter('alpha'),
            'epsilon': self._serialize_hyperparameter('epsilon')
        })
        return config
    
    
    # other solution
    def minimize(self, loss, var_list):
        grads = self.get_gradients(loss, var_list)
        updates = []
        prev_dir = tf.zeros_like(var_list)
        prev_grad = tf.zeros_like(var_list)

        for i, (grad, var) in enumerate(zip(grads, var_list)):
            prev_grad_i = tf.gather(prev_grad, i)
            prev_dir_i = tf.gather(prev_dir, i)

            # Compute the new search direction
            if i == 0:
                dir_i = -grad
            else:
                beta_i = tf.reduce_sum(grad * (grad - prev_grad_i)) / tf.reduce_sum(prev_dir_i * (grad - prev_grad_i))
                dir_i = -grad + beta_i * prev_dir_i

            # Compute the step size using line search
            step_size_i = self._line_search(loss, var, dir_i, grad)

            # Update the variable
            new_var_i = var + step_size_i * dir_i
            updates.append(tf.assign(var, new_var_i))

            # Update the previous gradient and search direction
            new_prev_grad_i = grad
            new_prev_dir_i = dir_i
            # update into existing tensor according to indices
            prev_grad = tf.tensor_scatter_nd_update(prev_grad, [[i]], [new_prev_grad_i])
            prev_dir = tf.tensor_scatter_nd_update(prev_dir, [[i]], [new_prev_dir_i])

        self._updates = updates
        return self._updates

    def _line_search(self, loss, var, direction, grad):
        step_size = tf.constant(1.0, dtype=tf.float32)
        loss_init = loss()

        while True:
            new_var = var + step_size * direction
            tf.keras.backend.update(var, new_var)
            loss_new = loss()

            if loss_new > loss_init + self.alpha * step_size * tf.reduce_sum(grad * direction):
                step_size *= self.beta
            else:
                break

        return step_size



class NonlinearCG_PRP_Wolfe(optimizer.Optimizer):
    """
    Nonlinear conjugate gradient optimizer with Polak-Ribière-Polyak version and Wolfe line search.
    """
    
    def __init__(self, alpha=1.0, beta=0.5, c1=1e-4, c2=0.9, **kwargs):
        """
        Constructor.
        
        Parameters:
            alpha (float): Initial step size.
            beta (float): Restart parameter.
            c1 (float): Armijo-Goldstein parameter.
            c2 (float): Curvature parameter.
        """
        super().__init__(**kwargs)
        self.alpha = alpha
        self.beta = beta
        self.c1 = c1
        self.c2 = c2
    
    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, 'prev_grad')
            self.add_slot(var, 'prev_direction')
            
    def _backtrack_wolfe(self, f, x, grad, direction, alpha, phi, phi_0, phi_prime_0, c1, c2, maxiter=10):
        """
        Perform a backtracking search to find the step size that satisfies the Wolfe conditions.
        
        Parameters:
            f (function): The loss function to optimize.
            x (tf.Variable): The current variable values.
            grad (tf.Tensor): The gradient of the loss with respect to the variables.
            direction (tf.Tensor): The search direction.
            alpha (float): The current step size.
            phi (float): The value of the loss function at x + alpha * direction.
            phi_0 (float): The value of the loss function at x.
            phi_prime_0 (float): The value of the directional derivative of the loss at x.
            c1 (float): Armijo-Goldstein parameter.
            c2 (float): Curvature parameter.
            maxiter (int): The maximum number of iterations to perform.
            
        Returns:
            float: The step size that satisfies the Wolfe conditions.
        """
        for i in range(maxiter):
            alpha *= 0.5
            x_next = x + alpha * direction
            phi = f(x_next)
            phi_prime = tf.reduce_sum(grad(x_next) * direction)
            
            # Armijo-Goldstein condition
            if phi > phi_0 + c1 * alpha * phi_prime_0:
                return self._backtrack_wolfe(f, x, grad, direction, alpha, phi, phi_0, phi_prime_0, c1, c2)
            
            # Curvature condition
            if phi_prime < c2 * phi_prime_0:
                return self._backtrack_wolfe(f, x, grad, direction, alpha, phi, phi_0, phi_prime_0, c1, c2)
            
        return alpha
    
    def _line_search_wolfe(self, f, x, grad, direction, c1, c2, maxiter=10):
        """
        Perform a line search to find the step size that satisfies the Wolfe conditions.
        
        Parameters:
            f (function): The loss function to optimize.
            x (tf.Variable): The current variable values.
            grad (tf.Tensor): The gradient of the loss with respect to the variables.
            direction (tf.Tensor): The search direction.
            c1 (float): Armijo-Goldstein parameter.
            c2 (float): Curvature parameter.
            maxiter (int): The maximum number of iterations to perform.
            
        Returns:
            float: The step size that satisfies the Wolfe conditions.
        """
        alpha = 1.0
        phi_0 = f(x)
        phi_prev = phi_0
        phi_prime_0 = tf.reduce_sum(grad * direction)
        
        for i in range(maxiter):
            x_next = x + alpha * direction
            phi = f(x_next)
            phi_prime = tf.reduce_sum(grad(x_next) * direction)
            
            # Armijo-Goldstein condition
            if phi > phi_0 + c1 * alpha * phi_prime_0 or (phi >= phi_prev and i > 0):
                return self._backtrack_wolfe(f, x, grad, direction, alpha, phi, phi_0, phi_prime_0, c1, c2)
            
            # Curvature condition
            if phi_prime < c2 * phi_prime_0:
                return self._backtrack_wolfe(f, x, grad, direction, alpha, phi, phi_0, phi_prime_0, c1, c2)
            
            phi_prev = phi
            alpha *= 2.0
        
        return alpha
    
    @tf.function
    def _resource_apply_dense(self, grad, var):
        prev_grad = self.get_slot(var, 'prev_grad')
        prev_direction = self.get_slot(var, 'prev_direction')
        
        # Compute direction
        if prev_grad is None or prev_direction is None:
            direction = -grad
        else:
            beta_PRP = tf.reduce_sum((grad - prev_grad) * grad) / tf.reduce_sum(prev_grad * prev_grad)
            beta = tf.maximum(beta_PRP, 0.0)
            direction = -grad + beta * prev_direction
        
        # Perform line search
        f = lambda x: (self._optimizer._loss(x), grad, var)
        alpha = self._line_search_wolfe(f, var, grad, direction, self.c1, self.c2)
        
        # Update variables and slots
        var.assign_add(alpha * direction)
        self.get_slot(var, 'prev_grad').assign(grad)
        self.get_slot(var, 'prev_direction').assign(direction)
    
    def _resource_apply_sparse(self, grad, var, indices):
        raise NotImplementedError("Sparse gradients are not supported.")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'alpha': self.alpha,
            'beta': self.beta,
            'c1': self.c1,
            'c2': self.c2,
        })
        return config
