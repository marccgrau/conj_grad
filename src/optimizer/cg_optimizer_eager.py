import tensorflow as tf
import warnings
import numpy as np
import logging

from tensorflow.keras import backend as K
from src.utils.custom import as_Kfloat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NonlinearCGEager(tf.keras.optimizers.Optimizer):
    def __init__(self, model, loss, max_iters=4, tol=1e-7, c1=1e-4, c2=0.1, amax=1.0, name='NLCG', **kwargs):
        super().__init__(name, **kwargs)
        self.max_iters = max_iters
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        self.amax = amax
        self.model = model
        self.loss = loss
        # function call counters
        self.objective_tracker = 0
        self.grad_tracker = 0
        # model specifics
        self._weight_shapes = tf.shape_n(self.model.trainable_variables)
        self._n_weights = len(self._weight_shapes)
        self._weight_indices = []
        self._weight_partitions = []
        
        param_count = 0
        for i, shape in enumerate(self._weight_shapes):
            n_params = tf.reduce_prod(shape)
            self._weight_indices.append(
                tf.reshape(
                    tf.range(param_count, param_count + n_params, dtype=tf.int32), shape
                )
            )
            self._weight_partitions.extend(
                tf.ones(shape=(n_params,), dtype=tf.int32) * i
            )
            param_count += n_params
        
        self.weights = self._from_matrices_to_vector(model.weights)


    
    @tf.function
    def _from_vector_to_matrices(self, vector):
        return tf.dynamic_partition(
            data=vector,
            partitions=self._weight_partitions,
            num_partitions=self._n_weights,
        )

    @tf.function
    def _from_matrices_to_vector(self, matrices):
        return tf.dynamic_stitch(indices=self._weight_indices, data=matrices)

    @tf.function
    def _update_model_parameters(self, new_params):
        params = self._from_vector_to_matrices(new_params)
        for i, (param, shape) in enumerate(zip(params, self._weight_shapes)):
            param = tf.reshape(param, shape)
            param = tf.cast(param, dtype=K.floatx())
            self.model.trainable_variables[i].assign(param)

    @tf.function
    def _objective_call(self, weights, x, y):
        # 1D Tensor to model weights and set for model
        self._update_model_parameters(weights)
        self.objective_tracker += 1
        return self.loss(y, self.model(x, training=True))
    
    @tf.function
    def _gradient_call(self, weights, x, y):
        # 1D Tensor to model weights and set for model
        self._update_model_parameters(weights)
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss_value = self.loss(y, y_pred)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.grad_tracker += 1
        self.objective_tracker += 1
        return self._from_matrices_to_vector(grads)
    
    @tf.function
    def _obj_func_and_grad_call(self, weights, x, y):
        self._update_model_parameters(weights)
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss_value = self.loss(y, y_pred)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.grad_tracker += 1
        self.objective_tracker += 1
        return loss_value, self._from_matrices_to_vector(grads)
    
    @tf.function
    def _save_new_model_weights(self, weights) -> None:
        self.weights = weights
        self._update_model_parameters(weights)
    
    @tf.function
    def update_step(self, x, y):
        iters = 0 
        obj_val, grad = self._obj_func_and_grad_call(self.weights, x, y)
        r = -grad
        d = r
        while iters < self.max_iters:
            # Perform line search to determine alpha_star
            alpha = self.wolfe_line_search(maxiter=3, search_direction=d, x=x, y=y)
            logger.info(f'alpha after line search: {alpha}')
            # update weights along search directions
            if alpha is None:
                logger.warning("Line search did not converge. Stopping optimization.")
                w_new = self.weights + 10e-2 * d
                self._save_new_model_weights(w_new)
                break
            else:
                w_new = self.weights + alpha * d
            # get new objective value and gradient
            obj_val_new, grad_new = self._obj_func_and_grad_call(w_new, x, y)
            # set r_{k+1}
            r_new = -grad_new
            # Calculate Polak-Ribiére beta
            beta = tf.reduce_sum(tf.multiply(r_new, r_new - r)) / tf.reduce_sum(tf.multiply(r, r))
            # PRP+ with max{beta{PR}, 0}
            logger.info(f'beta: {beta}')
            beta = np.maximum(beta, 0)
            # Determine new search direction for next iteration step
            d_new = r_new + beta * d
            # Check for convergence
            if tf.reduce_sum(tf.abs(obj_val_new - obj_val)) < self.tol:
                break
            tf.print("\n Iteration: ", iters, "Objective Value: ", obj_val_new)
            # Store new weights and set them for the model
            self._save_new_model_weights(w_new)
            # Set new values as old values for next iteration step
            grad = grad_new
            r = r_new
            d = d_new
            obj_val = obj_val_new
            iters += 1
    
    @tf.function
    def apply_gradients(self, vars, x,y):
        self.update_step(x, y)
        self._update_model_parameters(self.weights)
        return self.model
        
    @tf.function
    def wolfe_line_search(self, maxiter=10, search_direction=None, x=None, y=None):
        """
        Find alpha that satisfies strong Wolfe conditions.
        alpha > 0 is assumed to be a descent direction. #NOTE: Not always the case for Polak-Ribiere
        Parameters
        ----------
        c1: float, optional
            Parameter for Armijo condition rule.
        c2: float, optional
            Parameter for curvature condition rule.
        amax: float, optional
            Maximum step size.
        maxiter: int, optional
            Maximum number of iterations.
        search_direction: tf.Tensor, optional
            Search direction for line search determined by previous iteration step.
        Returns
        -------
        alpha_star: float or None
            Best alpha, or None if the line search algorithm did not converge.
        """
        
        # Leaving the weights as is, is the equivalent of setting alpha to 0
        # Thus, we get objective value at 0 and the gradient at 0        
        phi0 = self._objective_call(self.weights, x, y)
        # We need the directional derivative at 0 following the Wolfe Conditions
        # Thus, we get the gradient at 0 and multiply it with the search direction
        derphi0 = tf.tensordot(self._gradient_call(self.weights, x, y), search_direction, 1)
        
        # Set alpha bounds
        alpha0 = 0.
        alpha1 = 1.
        # Optional setting of an alpha max, if defined
        if self.amax is not None:
            alpha1 = np.minimum(alpha1, self.amax)
        
        # get objective value at a new possible position, i.e. w_k + alpha1 * d_k
        phi_a1 = self._objective_call(self.weights + alpha1 * search_direction, x, y)
        
        # Initial alpha_lo equivalent to alpha = 0
        phi_a0 = phi0
        derphi_a0 = derphi0
        i = 0
        for i in range(maxiter):
            #tf.cond(self.initial_cond(), lambda: break_ = True, lambda: None)                
            if alpha1 == 0. or alpha0 == self.amax:
            #if tf.math.logical_or(tf.math.equal(alpha1, tf.Variable(0.)), tf.math.equal(alpha0, self.amax)):
                alpha_star = None
                #phi_star = phi0
                #derphi_star = None
                
                if alpha1 == 0:
                #if tf.math.equal(alpha1, 0):
                    warnings.warn('Rounding errors preventing line search from converging')
                else:
                    warnings.warn(f'Line search could not find solution less than or equal to {self.amax}')
                break
            
            # First condition: phi(alpha_i) > phi(0) + c1 * alpha_i * phi'(0) or [phi(alpha_i) >= phi(alpha_{i-1}) and i > 1]
            if phi_a1 > phi0 + self.c1 * alpha1 * derphi0 or (phi_a1 >= phi_a0 and i > 1):
            #if tf.math.logical_or(tf.math.greater(phi_a1, phi0 + self.c1 * alpha1 * derphi0), tf.math.logical_and(tf.math.greater_equal(phi_a1, phi_a0), tf.math.greater(tf.Variable(i), 1))):
                alpha_star = self._zoom(alpha0,
                                        alpha1,
                                        phi_a0,
                                        phi_a1, 
                                        derphi_a0,
                                        phi0, 
                                        derphi0, 
                                        search_direction,
                                        x,
                                        y,
                                        )
                break
            
            # Second condition: |phi'(alpha_i)| <= -c2 * phi'(0)
            derphi_a1 = tf.tensordot(self._gradient_call(self.weights + alpha1 * search_direction, x, y), search_direction, 1)
            if np.absolute(derphi_a1) <= -self.c2 * derphi0:
            #if tf.math.lower_equal(tf.math.abs(derphi_a1), -self.c2 * derphi0):
                alpha_star = alpha1 # suitable alpha found set to star and return
                phi_star = self._objective_call(self.weights + alpha_star * search_direction, x, y)
                derphi_star = derphi_a1
                break
            
            # Third condition: phi'(alpha_i) >= 0
            if derphi_a1 >= 0.:
            #if (tf.math.greater_equal(derphi_a1, 0)):
                alpha_star = self._zoom(alpha1,
                                        alpha0,
                                        phi_a1,
                                        phi_a0,
                                        derphi_a1,
                                        phi0,
                                        derphi0,
                                        search_direction,
                                        x,
                                        y,
                                        )
                break
            
            # extrapolation step of alpha_i as no conditions are met
            # simple procedure to mulitply alpha by 2
            alpha2 = 2*alpha1
            # check if we don't overshoot amax
            if self.amax is not None:
                alpha2 = np.minimum(alpha2, self.amax)
                #alpha2 = tf.math.minimum(alpha2,self.amax)
            
            # update values for next iteration to check conditions
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            phi_a1 = self._objective_call(self.weights + alpha1 * search_direction, x, y)
            derphi_a0 = derphi_a1
        
        # if no break occurs, then we have not found a suitable alpha after maxiter
        else:
            alpha_star = alpha1
            warnings.warn('Line search did not converge')
        
        return alpha_star
    
    @tf.function
    def _zoom(self, a_lo, a_hi, phi_lo, phi_hi, derphi_lo, phi0, derphi0, search_direction, x, y):
        """
        Zoom stage of approximate line search satisfying strong Wolfe conditions.
        """

        maxiter = 10
        i = 0
        delta1 = 0.2  # cubic interpolant check
        delta2 = 0.1  # quadratic interpolant check
        phi_rec = phi0
        a_rec = 0
        while True:
            # interpolate to find a trial step length between a_lo and
            # a_hi Need to choose interpolation here. Use cubic
            # interpolation and then if the result is within delta *
            # dalpha or outside of the interval bounded by a_lo or a_hi
            # then use quadratic interpolation, if the result is still too
            # close, then use bisection

            dalpha = a_hi - a_lo
            if dalpha < 0:
                a, b = a_hi, a_lo
            else:
                a, b = a_lo, a_hi

            # minimizer of cubic interpolant
            # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
            #
            # if the result is too close to the end points (or out of the
            # interval), then use quadratic interpolation with phi_lo,
            # derphi_lo and phi_hi if the result is still too close to the
            # end points (or out of the interval) then use bisection

            if (i > 0):
                cchk = delta1 * dalpha
                a_j = self._cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi,
                                a_rec, phi_rec)
            if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
                qchk = delta2 * dalpha
                a_j = self._quadmin(point_1=a_lo, 
                                    obj_1=phi_lo, 
                                    grad_1=derphi_lo, 
                                    point_2=a_hi, 
                                    obj_2=phi_hi)
                if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                    a_j = a_lo + 0.5*dalpha

            # Check new value of a_j
            phi_aj = self._objective_call(self.weights + tf.math.scalar_mul(as_Kfloat(a_j),search_direction), x, y)
            if (phi_aj > phi0 + self.c1*a_j*derphi0) or (phi_aj >= phi_lo):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj
            else:
                derphi_aj = tf.tensordot(self._gradient_call(self.weights + a_j * search_direction, x, y), search_direction, 1)
                if tf.math.abs(derphi_aj) <= -self.c2*derphi0:
                    a_star = a_j
                    break
                if derphi_aj*(a_hi - a_lo) >= 0:
                    phi_rec = phi_hi
                    a_rec = a_hi
                    a_hi = a_lo
                    phi_hi = phi_lo
                else:
                    phi_rec = phi_lo
                    a_rec = a_lo
                a_lo = a_j
                phi_lo = phi_aj
                derphi_lo = derphi_aj
            i += 1
            if (i > maxiter):
                # Failed to find a conforming step size
                a_star = None
                break
        return a_star #, val_star, valprime_star

    @tf.function
    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        """
        Finds the minimizer for a cubic polynomial that goes through the
        points (a,fa), (b,fb), and (c,fc) with derivative at a of fpa.
        If no minimizer can be found, return None.
        """
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = np.dot(d1, np.asarray([fb - fa - C * db, fc - fa - C * dc], dtype='float64').flatten())
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
        if not np.isfinite(xmin):
            return None
        return xmin
    
    @tf.function
    def _quadmin(self,point_1, obj_1, grad_1, point_2, obj_2):
        """
        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at a of fpa.
        """
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        try:
            D = obj_1
            C = grad_1
            db = point_2 - point_1 * 1.0
            B = (obj_2 - D - C * db) / (db * db)
            xmin = point_1 - C / (2.0 * B)
        except ArithmeticError:
            return None
        if not np.isfinite(xmin):
            return None
        return xmin
    
    def get_config(self):
        pass


