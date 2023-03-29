import torch
import numpy as np


class NonlinearCG(torch.optim.Optimizer):
    def __init__(self, model, loss, max_iters=1000, tol=1e-7, c1=1e-4, c2=0.1, amax=1.0, name='NonlinearCG', **kwargs):
        super(NonlinearCG, self).__init__(name, **kwargs)
        self.max_iters = max_iters
        self.tol = tol
        self.c1 = c1
        self.c2 = c2
        self.amax = amax
        self.model = model
        self.weights = self._pack_weights(model.trainable_variables)
        self.loss = loss
        self.objective_tracker = 0
        self.grad_tracker = 0
        
    # pack model into 1D tensor
    def _pack_weights(self, weights) -> torch.Tensor:
        return torch.concat([torch.reshape(g, [-1]) for g in weights], axis=0)

    # unpack model from 1D tensor
    def _unpack_weights(self, packed_weights):
        i = 0
        unpacked = []
        for layer in self.model.layers:
            for current in layer.weights:
                length = torch.prod(current.shape)
                unpacked.append(
                    torch.reshape(packed_weights[i : i + length], current.shape)
                )
                i += length
        return unpacked

    def _objective_call(self, weights, x, y):
        # 1D Tensor to model weights and set for model
        #self.model.set_weights(self._unpack_weights(weights))
        self.objective_tracker += 1
        return self.loss(y, self.model(x))
    
    def _gradient_call(self, weights, x, y):
        # 1D Tensor to model weights and set for model
        #self.model.set_weights(self._unpack_weights(weights))
        with tf.GradientTape() as tape:
            loss_value = self.loss(y, self.model(x))
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.grad_tracker += 1
        self.objective_tracker += 1
        return self._pack_weights(grads)
    
    def _obj_func_and_grad_call(self, weights, x, y):
        #self.model.set_weights(self._unpack_weights(weights))
        with tf.GradientTape() as tape:
            loss_value = self.loss(y, self.model(x))
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.grad_tracker += 1
        self.objective_tracker += 1
        return loss_value, self._pack_weights(grads)
    
    # Crucial, final update of model weights after optimization
    # Final status of iteration step for both weights and model
    def _save_new_model_weights(self, weights) -> None:
        self.weights = weights
        self.model.set_weights(self._unpack_weights(weights))
        
    def update_step(self, x, y):
        iters = 0 
        obj_val, grad = self._obj_func_and_grad_call(self.weights, x, y)
        r = -grad
        d = r
        while iters < self.max_iters:
            # Perform line search to determine alpha_star
            alpha = self.wolfe_line_search(maxiter=tf.Variable(10), search_direction=d, x=x, y=y)
            # update weights along search directions
            w_new = self.weights + alpha * d
            # get new objective value and gradient
            obj_val_new, grad_new = self._obj_func_and_grad_call(w_new, x, y)
            # set r_{k+1}
            r_new = -grad_new
            # Calculate Polak-Ribiére beta
            beta = tf.reduce_sum(tf.multiply(r_new, r_new - r)) / tf.reduce_sum(tf.multiply(r, r))
            # Determine new search direction for next iteration step
            d_new = r_new + beta * d
            # Check for convergence
            if tf.reduce_sum(tf.abs(obj_val_new - obj_val)) < self.tol:
                break
            tf.print("Iteration: ", iters, "Objective Value: ", obj_val_new)
            # Store new weights and set them for the model
            self._save_new_model_weights(w_new)
            # Set new values as old values for next iteration step
            grad = grad_new
            r = r_new
            d = d_new
            obj_val = obj_val_new
            iters += 1
    
    def apply_gradients(self, vars, x,y):
        self.update_step(x, y)
        for var in vars:
            var.assign(self._unpack_weights(self.model.weights))
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
        alpha_star = tf.Variable(0, dtype='float32')
        # Leaving the weights as is, is the equivalent of setting alpha to 0
        # Thus, we get objective value at 0 and the gradient at 0        
        phi0 = self._objective_call(self.weights, x, y)
        # We need the directional derivative at 0 following the Wolfe Conditions
        # Thus, we get the gradient at 0 and multiply it with the search direction
        derphi0 = self._gradient_call(self.weights, x, y) * search_direction
        
        # Set alpha bounds
        alpha0 = tf.Variable(0, dtype='float32')
        alpha1 = tf.Variable(1, dtype='float32')
        # Optional setting of an alpha max, if defined
        if self.amax is not None:
            alpha1 = tf.math.minimum(alpha1, self.amax)
        
        # get objective value at a new possible position, i.e. w_k + alpha1 * d_k
        phi_a1 = self._objective_call(self.weights + alpha1 * search_direction, x, y)
        derphi_a1 = self._gradient_call(self.weights + alpha1 * search_direction, x, y) * search_direction
        
        # Initial alpha_lo equivalent to alpha = 0
        phi_a0 = phi0
        derphi_a0 = derphi0
        
        @tf.function
        def intitial_cond(alpha1, alpha0):
            alpha_star.assign(0)
            break_.assign(True)
            
            tf.cond(
                tf.math.logical_or(tf.math.equal(alpha1, tf.Variable(0, dtype='float32')), tf.math.equal(alpha0, self.amax)),
                true_fn = break_.assign(True),
                false_fn = break_.assign(False),
            )
            
            return alpha_star, break_
        
        @tf.function
        def first_cond(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi0, derphi0, search_direction, x, y, i):
            alpha_star.assign(0)
            break_.assign(True)
            
            tf.cond(
                tf.math.logical_or(tf.math.greater(phi_a1, phi0 + self.c1 * alpha1 * derphi0), tf.math.logical_and(tf.math.greater_equal(phi_a1, phi_a0), tf.math.greater(i, 1))),
                true_fn = alpha_star.assign(self._zoom(alpha0,
                                                        alpha1,
                                                        phi_a0,
                                                        phi_a1, 
                                                        derphi_a0,
                                                        phi0, 
                                                        derphi0, 
                                                        search_direction,
                                                        x,
                                                        y,
                                                        )),
                false_fn = break_.assign(False),
            )
            
            return alpha_star, break_
        
        @tf.function
        def second_cond(alpha1, derphi_a1, derphi0):
            alpha_star.assign(0)
            break_.assign(True)
            
            tf.cond(
                tf.math.less_equal(tf.math.abs(derphi_a1), -self.c2 * derphi0),
                true_fn=alpha_star.assign(alpha1),
                false_fn=break_.assign(False),
            )
            
            return alpha_star, break_
        
        @tf.function
        def third_cond(alpha0, alpha1, phi_a0, phi_a1, derphi_a1, phi0, derphi0, search_direction, x, y):
            alpha_star.assign(0)
            break_.assign(True)
            
            tf.cond(
                tf.math.greater_equal(derphi_a1, 0),
                true_fn = alpha_star.assign(self._zoom(alpha1,
                                                        alpha0,
                                                        phi_a1,
                                                        phi_a0,
                                                        derphi_a1,
                                                        phi0,
                                                        derphi0,
                                                        search_direction,
                                                        x,
                                                        y,
                                                    )),
                false_fn = break_.assign(False)
            )
            
            return alpha_star, break_
        
        '''
        @tf.function
        def while_cond(i, break_, alpha_star):
            return tf.logical_and(tf.less(i, maxiter), tf.math.logical_not(break_), tf.math.less_equal(alpha_star, self.amax))
        '''
        
        cond = lambda i, break_, alpha_star: tf.logical_and(tf.less(i, maxiter), tf.math.logical_not(break_), tf.math.less_equal(alpha_star, self.amax))
        
        @tf.function
        def while_body():
            alpha_star, break_ = intitial_cond(alpha1, alpha0)
            alpha_star, break_ = first_cond(alpha0, alpha1, phi_a0, phi_a1, derphi_a0, phi0, derphi0, search_direction, x, y, i)
            alpha_star, break_ = second_cond(derphi_a1, derphi0)
            alpha_star, break_ = third_cond(alpha0, alpha1, phi_a0, phi_a1, derphi_a1, phi0, derphi0, search_direction, x, y)
            
            # extrapolation step of alpha_i as no conditions are met
            # simple procedure to mulitply alpha by 2
            alpha2 = 2*alpha1
            # check if we don't overshoot amax
            if self.amax is not None:
                alpha2 = tf.math.minimum(alpha2, self.amax)
            
            # update values for next iteration to check conditions
            alpha0 = alpha1
            alpha1 = alpha2
            phi_a0 = phi_a1
            phi_a1 = self._objective_call(self.weights + alpha1 * search_direction, x, y)
            derphi_a0 = derphi_a1
            
            return i+1, break_, alpha_star     
            
        
        # Initiate line search by checking for the three conditions as defined in paper
        break_ = tf.Variable(False, dtype='bool')
        i = tf.Variable(0, dtype='int32')
        result = tf.while_loop(cond=cond, body=while_body, loop_vars=[0, False, 0.])
        return result['alpha_star']
        
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
                a_j = self._quadmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi)
                if (a_j is None) or (a_j > b-qchk) or (a_j < a+qchk):
                    a_j = a_lo + 0.5*dalpha

            # Check new value of a_j

            phi_aj = self._objective_call(self.weights + a_j * search_direction, x, y)
            if (phi_aj > phi0 + self.c1*a_j*derphi0) or (phi_aj >= phi_lo):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj
            else:
                derphi_aj = self._gradient_call(self.weights + a_j * search_direction, x, y) * search_direction
                if tf.math.abs(derphi_aj) <= -self.c2*derphi0:
                    a_star = a_j
                    val_star = phi_aj
                    valprime_star = derphi_aj
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
                val_star = None
                valprime_star = None
                break
        return a_star #, val_star, valprime_star

    def _cubicmin(a, fa, fpa, b, fb, c, fc):
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
            d1 = tf.reshape((), (2, 2))
            d1[0, 0] = dc ** 2
            d1[0, 1] = -db ** 2
            d1[1, 0] = -dc ** 3
            d1[1, 1] = db ** 3
            [A, B] = tf.tensordot(d1, tf.Tensor([fb - fa - C * db,
                                            fc - fa - C * dc]).reshape([-1]))
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + tf.math.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
        if not tf.math.isfinite(xmin):
            return None
        return xmin
    
    def _quadmin(a, fa, fpa, b, fb):
        """
        Finds the minimizer for a quadratic polynomial that goes through
        the points (a,fa), (b,fb) with derivative at a of fpa.
        """
        # f(x) = B*(x-a)^2 + C*(x-a) + D
        try:
            D = fa
            C = fpa
            db = b - a * 1.0
            B = (fb - D - C * db) / (db * db)
            xmin = a - C / (2.0 * B)
        except ArithmeticError:
            return None
        if not tf.math.isfinite(xmin):
            return None
        return xmin
    
    def get_config(self):
        pass


