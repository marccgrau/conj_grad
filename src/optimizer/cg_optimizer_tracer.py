import tensorflow as tf
import warnings
import numpy as np
import logging

from tensorflow.keras import backend as K
from src.utils.custom import as_Kfloat

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NonlinearCG(tf.keras.optimizers.Optimizer):
    def __init__(
        self,
        model,
        loss,
        max_iters=10,
        tol=1e-7,
        c1=1e-4,
        c2=0.9,
        amax=1.0,
        name="NLCG",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        # general variables
        self.max_iters = tf.Variable(max_iters, name="max_iters", dtype=tf.float64)
        self.tol = tf.Variable(tol, name="tolerance")
        self.c1 = tf.Variable(c1, name="c1")
        self.c2 = tf.Variable(c2, name="c2")
        self.amax = tf.Variable(amax, name="amax")
        self.alpha = tf.Variable(initial_value=0.0, name="alpha", dtype=tf.float64)
        self.model = model
        self.loss = loss
        # function call counters
        self.objective_tracker = tf.Variable(0, name="objective_tracker")
        self.grad_tracker = tf.Variable(0, name="gradient_tracker")
        # model specifics
        self._weight_shapes = tf.shape_n(self.model.trainable_variables)
        self._n_weights = len(self._weight_shapes)
        self._weight_indices = []
        self._weight_partitions = []
        # optimization variables
        self.alpha_star = tf.Variable(0.0, name="alpha_star", dtype=tf.float64)
        self.alpha0 = tf.Variable(0.0, name="alpha0", dtype=tf.float64)
        self.alpha1 = tf.Variable(1.0, name="alpha1", dtype=tf.float64)
        self.alpha2 = tf.Variable(1.0, name="alpha2", dtype=tf.float64)
        self.phi_star = tf.Variable(0.0, name="phi_star", dtype=tf.float64)
        self.phi0 = tf.Variable(0.0, name="phi0", dtype=tf.float64)
        self.phi_a0 = tf.Variable(0.0, name="phi_a0", dtype=tf.float64)
        self.phi_a1 = tf.Variable(0.0, name="phi_a1", dtype=tf.float64)
        self.derphi_star = tf.Variable(0, name="derphi_star", dtype=tf.float64)
        self.derphi0 = tf.Variable(0, name="derphi0", dtype=tf.float64)
        self.derphi_a0 = tf.Variable(0, name="derphi_a0", dtype=tf.float64)
        self.derphi_a1 = tf.Variable(0, name="derphi_a1", dtype=tf.float64)
        self._break = tf.Variable(False, dtype=bool)
        self._update_step_break = tf.Variable(False, dtype=bool)
        # zoom function
        self.delta1 = 0.2  # cubic interpolant check
        self.delta2 = 0.1  # quadratic interpolant check

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

        self.weights = tf.Variable(
            self._from_matrices_to_vector(model.trainable_variables), name="weights"
        )

    @tf.function
    def _from_vector_to_matrices(self, vector):
        """
        Turn 1D weight representation into model representation
        Parameters
        ----------
        vector: tf.Tensor
            1D representation of weights
        Returns
        -------
        dynamic_partition: tf.Tensor
            Weights in model representation
        """
        return tf.dynamic_partition(
            data=vector,
            partitions=self._weight_partitions,
            num_partitions=self._n_weights,
        )

    @tf.function
    def _from_matrices_to_vector(self, matrices: tf.Tensor):
        """
        Turn weights in model representation to 1D representation
        Parameters
        ----------
        matrices: tf.Tensor
            Weights in model representation
        Returns
        -------
        vector: tf.Tensor
            1D representation of weights
        """
        return tf.dynamic_stitch(indices=self._weight_indices, data=matrices)

    @tf.function
    def _update_model_parameters(self, new_params: tf.Tensor):
        """
        Assign new set of weights to model
        Parameters
        ----------
        new_params: tf.Tensor
            New model weights
        Returns
        -------

        """
        params = self._from_vector_to_matrices(new_params)
        for i, (param, shape) in enumerate(zip(params, self._weight_shapes)):
            param = tf.reshape(param, shape)
            param = tf.cast(param, dtype=K.floatx())
            self.model.trainable_variables[i].assign(param)

    @tf.function
    def _objective_call(self, weights: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
        """
        Calculate value of objective function given a certain set of model weights
        Add +1 to tracker of objective function calls
        Parameters
        ----------
        weights: tf.Tensor
            Set of possible model weights
        x: tf.Tensor
        y: tf.Tensor
        Returns
        -------
        loss: tf.Tensor
        """
        self._update_model_parameters(weights)
        self.objective_tracker.assign_add(1)
        return self.loss(y, self.model(x, training=True))

    @tf.function
    def _gradient_call(self, weights: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
        """
        Calculate value of objective function given a certain set of model weights
        Calculate gradient for the given set of model weights
        Update both objective function and gradient call trackers
        Return only gradient
        Parameters
        ----------
        weights: tf.Tensor
            Set of possible model weights
        x: tf.Tensor
        y: tf.Tensor
        Returns
        -------
        grads: tf.Tensor in 1D representation
        """
        self._update_model_parameters(weights)
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss_value = self.loss(y, y_pred)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.grad_tracker.assign_add(1)
        self.objective_tracker.assign_add(1)
        return self._from_matrices_to_vector(grads)

    @tf.function
    def _obj_func_and_grad_call(self, weights: tf.Tensor, x: tf.Tensor, y: tf.Tensor):
        """
        Calculate value of objective function given a certain set of model weights
        Calculate gradient for the given set of model weights
        Update both objective function and gradient call trackers
        Returns both loss and gradient
        Parameters
        ----------
        weights: tf.Tensor
            Set of possible model weights
        x: tf.Tensor
        y: tf.Tensor
        Returns
        -------
        loss: tf.Tensor
        grads: tf.Tensor in 1D representation
        """
        self._update_model_parameters(weights)
        with tf.GradientTape() as tape:
            y_pred = self.model(x, training=True)
            loss_value = self.loss(y, y_pred)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.grad_tracker.assign_add(1)
        self.objective_tracker.assign_add(1)
        return loss_value, self._from_matrices_to_vector(grads)

    @tf.function
    def _save_new_model_weights(self, weights: tf.Tensor) -> None:
        """
        Get new set of weights and assign it to the model
        Parameters
        ----------
        weights: tf.Tensor
            Set of possible model weights
        Returns
        -------

        """
        self.weights.assign(weights)
        self._update_model_parameters(weights)

    def wolfe_and_conj_grad_step(iters, maxiter, d, r, obj_val, x, y):
        self.wolfe_line_search(maxiter=maxiter, search_direction=d, x=x, y=y)
        tf.print(self.alpha)
        d, r, obj_val = self.conj_grad_step(self.alpha, d, r, obj_val, x, y)
        iters = tf.add(iters, 1)
        return iters, d, r, obj_val

    def update_step(self, x: tf.Tensor, y: tf.Tensor):
        """
        Initialise conjugate gradient method by setting initial search direction
        Nonlinear conjugate gradient optimization procedure with line search satisfying Wolfe conditions
        Beta of Polak-Ribière with max(beta, 0) to satisfy convergence conditions

        Parameters
        ----------
        x: tf.Tensor
        y: tf.Tensor
        Returns
        -------

        """
        # iters = tf.constant(0)
        obj_val, grad = self._obj_func_and_grad_call(self.weights, x, y)
        r = -grad
        d = r
        # cond = lambda iters, d, r, obj_val: tf.less(iters, 10)
        # body = lambda iters, d, r, obj_val: self.wolfe_and_conj_grad_step(iters=iters, maxiter=10, d=d, r=r, obj_val=obj_val)
        # iters, d, r, obj_val = tf.while_loop(cond=cond, body=body, loop_vars=[iters, d, r, obj_val])
        for i in range(10):
            # Perform line search to determine alpha_star
            self.wolfe_line_search(maxiter=10, search_direction=d, x=x, y=y)
            logger.info(f"alpha after line search: {self.alpha}")
            d, r, obj_val = self.conj_grad_step(self.alpha, d, r, obj_val, x, y)
            if self.alpha == 0:
                break

    @tf.function
    def conj_grad_step(self, alpha, d, r, obj_val, x, y):
        if alpha == 0.0:
            tf.print("Alpha is zero. Making no step.")
            # w_new = self.weights + 10e-1 * r
            # self._save_new_model_weights(w_new)
            return d, r, obj_val
        else:
            w_new = self.weights + alpha * d
            self._save_new_model_weights(w_new)
        # get new objective value and gradient
        # NOTE: actually not necessary to assign params to model already done in step before
        obj_val_new, grad_new = self._obj_func_and_grad_call(self.weights, x, y)
        # set r_{k+1}
        r_new = -grad_new
        # Calculate Polak-Ribiére beta
        beta = tf.reduce_sum(tf.multiply(r_new, r_new - r)) / tf.reduce_sum(
            tf.multiply(r, r)
        )
        # PRP+ with max{beta{PR}, 0}
        beta = tf.math.maximum(beta, 0)
        tf.print(f"beta: {beta}")
        # Determine new search direction for next iteration step
        d_new = r_new + beta * d
        # Check for convergence
        if tf.reduce_sum(tf.abs(obj_val_new - obj_val)) < self.tol:
            tf.print(f"Stop NLCG, no sufficient decrease in value function")
            return d_new, r_new, obj_val_new
        # check if vector norm is smaller than the tolerance
        if tf.norm(r_new) <= self.tol:
            tf.print(f"Stop NLCG, gradient norm smaller than tolerance")
            return d_new, r_new, obj_val_new
        return d_new, r_new, obj_val_new

    def apply_gradients(self, vars, x, y):
        """
        Do exactly one update step, which could include multiple iterations
        Necessary to define for training procedure
        Parameters
        ----------
        vars: trainable variables of model
        x: tf.Tensor
        y: tf.Tensor
        Returns
        -------
        model: tf.keras.Model
        Updated model with new weights after iterations
        """
        self.update_step(x, y)
        self._update_model_parameters(self.weights)
        return self.model.get_weights()

    def wolfe_line_search(self, maxiter=10, search_direction=None, x=None, y=None):
        """
        Find alpha that satisfies strong Wolfe conditions.
        alpha > 0 is assumed to be a descent direction. #NOTE: Not always the case for Polak-Ribiere
        Parameters
        ----------
        c1: float
            Parameter for Armijo condition rule.
        c2: float
            Parameter for curvature condition rule.
        amax: float
            Maximum step size.
        maxiter: int
            Maximum number of iterations.
        search_direction: tf.Tensor
            Search direction for line search determined by previous iteration step.
        Returns
        -------
        alpha_star: float or None
            Best alpha, or None if the line search algorithm did not converge.
        """

        # Leaving the weights as is, is the equivalent of setting alpha to 0
        # Thus, we get objective value at 0 and the gradient at 0
        self.phi0.assign(self._objective_call(self.weights, x, y))
        # We need the directional derivative at 0 following the Wolfe Conditions
        # Thus, we get the gradient at 0 and multiply it with the search direction
        self.derphi0.assign(
            tf.tensordot(self._gradient_call(self.weights, x, y), search_direction, 1)
        )

        # Set alpha bounds
        # alpha0 = tf.Variable(0.0, dtype='float64')
        # alpha1 = tf.Variable(1.0, dtype='float64')
        # Optional setting of an alpha max, if defined
        if self.amax is not None:
            self.alpha1.assign(tf.math.minimum(self.alpha1, self.amax))

        # get objective value at a new possible position, i.e. w_k + alpha1 * d_k
        self.phi_a1.assign(
            self._objective_call(self.weights + self.alpha1 * search_direction, x, y)
        )

        # Initialization of all variables for loop
        # Initial alpha_lo equivalent to alpha = 0
        self.phi_a0.assign(self.phi0)
        self.derphi_a0.assign(self.derphi0)
        i = tf.Variable(0, dtype=tf.float64)

        # While cond
        def while_cond(i: tf.Variable):
            return tf.math.logical_and(
                tf.math.less(i, self.max_iters),
                tf.math.equal(self._break, tf.constant(False)),
            )

        # Define loop body
        def body(i: tf.Variable):
            def init_cond():
                return tf.math.logical_or(
                    tf.math.equal(self.alpha1, tf.Variable(0, dtype=tf.float64)),
                    tf.math.equal(self.alpha0, self.amax),
                )

            # necessary for second check
            self.derphi_a1.assign(
                tf.tensordot(
                    self._gradient_call(
                        self.weights + self.alpha1 * search_direction, x, y
                    ),
                    search_direction,
                    1,
                )
            )

            def first_cond():
                return tf.math.logical_or(
                    tf.math.greater(
                        self.phi_a1, self.phi0 + self.c1 * self.alpha1 * self.derphi0
                    ),
                    tf.math.logical_and(
                        tf.math.greater_equal(self.phi_a1, self.phi_a0),
                        tf.math.greater(tf.Variable(i), 1),
                    ),
                )

            def first_check_action():
                self.alpha_star.assign(
                    self._zoom(
                        self.alpha0,
                        self.alpha1,
                        self.phi_a0,
                        self.phi_a1,
                        self.derphi_a0,
                        self.phi0,
                        self.derphi0,
                        search_direction,
                        x,
                        y,
                    )
                )
                self.alpha.assign(self.alpha_star)
                tf.add(i, 1)
                self._break.assign(tf.Variable(True))

            def second_cond():
                return tf.math.less_equal(
                    tf.math.abs(self.derphi_a1), -self.c2 * self.derphi0
                )

            def second_check_action():
                self.alpha_star.assign(self.alpha1)
                self.phi_star.assign(
                    self._objective_call(
                        self.weights + self.alpha_star * search_direction, x, y
                    )
                )
                self.derphi_star.assign(self.derphi_a1)
                self.alpha.assign(self.alpha_star)
                tf.add(i, 1)
                self._break.assign(tf.Variable(True))

            def third_cond():
                return tf.math.greater_equal(
                    self.derphi_a1, tf.Variable(0.0, dtype=tf.float64)
                )

            def third_check_action():
                self.alpha_star.assign(
                    self._zoom(
                        self.alpha1,
                        self.alpha0,
                        self.phi_a1,
                        self.phi_a0,
                        self.derphi_a1,
                        self.phi0,
                        self.derphi0,
                        search_direction,
                        x,
                        y,
                    )
                )
                self.alpha.assign(self.alpha_star)
                tf.add(i, 1)
                self._break.assign(tf.Variable(True))

            def false_action():
                self._break.assign(tf.constant(False))

            def final_cond(i):
                return tf.math.equal(i, self.max_iters)

            tf.cond(
                init_cond(),
                lambda: (
                    self.alpha_star.assign(tf.Variable(0, dtype=tf.float64)),
                    tf.add(i, 0),
                    self._break.assign(tf.Variable(False)),
                ),
                lambda: (
                    self.alpha_star.assign(tf.Variable(0, dtype=tf.float64)),
                    tf.add(i, 1),
                    self._break.assign(tf.Variable(True)),
                ),
            )
            tf.cond(
                first_cond(),
                lambda: first_check_action(),
                lambda: false_action(),
            )
            tf.cond(
                second_cond(),
                lambda: second_check_action(),
                lambda: false_action(),
            )
            tf.cond(
                third_cond(),
                lambda: third_check_action(),
                lambda: false_action(),
            )

            self.alpha2.assign(2 * self.alpha1)

            self.alpha2.assign(tf.math.minimum(self.alpha2, self.amax))

            self.alpha0.assign(self.alpha1)
            self.alpha1.assign(self.alpha2)
            self.phi_a0.assign(self.phi_a1)
            self.phi_a1.assign(
                self._objective_call(
                    self.weights + self.alpha1 * search_direction, x, y
                )
            )
            self.derphi_a0.assign(self.derphi_a1)

            tf.add(i, 1)
            # if no break occurs, then we have not found a suitable alpha after maxiter
            tf.cond(
                final_cond(i),
                lambda: (
                    self.alpha_star.assign(self.alpha1),
                    self.alpha.assign(self.alpha_star),
                    self._break.assign(tf.Variable(True)),
                ),
                lambda: (
                    self.alpha_star,
                    self.alpha,
                    self._break.assign(tf.Variable(False)),
                ),
            )
            return i

        # While loop
        tf.while_loop(while_cond, body, [i])

    def _zoom(
        self,
        a_lo,
        a_hi,
        phi_lo,
        phi_hi,
        derphi_lo,
        phi0,
        derphi0,
        search_direction,
        x,
        y,
    ):
        """
        Zoom stage of approximate line search satisfying strong Wolfe conditions.
        """
        # TODO: maxiter as argument
        maxiter = 20
        i = 0
        phi_rec = phi0
        a_rec = tf.Variable(0, dtype=tf.float64)

        while True:
            # interpolate to find a trial step length between a_lo and
            # a_hi Need to choose interpolation here. Use cubic
            # interpolation and then if the result is within delta *
            # dalpha or outside of the interval bounded by a_lo or a_hi
            # then use quadratic interpolation, if the result is still too
            # close, then use bisection

            dalpha = a_hi - a_lo
            dalpha_cond = dalpha < 0
            dalpha = tf.where(dalpha_cond, -dalpha, dalpha)
            a_lo, a_hi = tf.cond(
                tf.reduce_any(dalpha_cond), lambda: (a_hi, a_lo), lambda: (a_lo, a_hi)
            )

            def cond_trial_step(dalpha, a_hi, a_lo):
                return tf.less(dalpha, 0)

            def true_fn_trial_step(dalpha, a_hi, a_lo):
                return a_hi, a_lo

            def false_fn_trial_step(dalpha, a_hi, a_lo):
                return a_lo, a_hi

            def trial_step_length(dalpha, a_hi, a_lo):
                a, b = tf.cond(
                    cond_trial_step(dalpha, a_hi, a_lo),
                    lambda: true_fn_trial_step(dalpha, a_hi, a_lo),
                    lambda: false_fn_trial_step(dalpha, a_hi, a_lo),
                )
                return a, b

            a, b = trial_step_length(dalpha, a_hi, a_lo)

            # minimizer of cubic interpolant
            # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
            #
            # if the result is too close to the end points (or out of the
            # interval), then use quadratic interpolation with phi_lo,
            # derphi_lo and phi_hi if the result is still too close to the
            # end points (or out of the interval) then use bisection

            def cond_interpolation1(i):
                return tf.greater(i, 0)

            def cond_interpolation2(i, a_j, a, b, cchk):
                return tf.logical_or(
                    tf.equal(i, 0),
                    tf.logical_or(
                        tf.equal(a_j, tf.constant(np.nan, dtype=tf.float64)),
                        tf.logical_or(
                            tf.greater(a_j, b - cchk), tf.less(a_j, a + cchk)
                        ),
                    ),
                )

            def cond_interpolation3(a_j, a, b, qchk):
                return tf.logical_or(
                    tf.equal(a_j, tf.constant(np.nan, dtype=tf.float64)),
                    tf.logical_or(tf.greater(a_j, b - qchk), tf.less(a_j, a + qchk)),
                )

            def true_fn_interpolation1(
                a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
            ):
                return self._cubicmin(
                    a=a_lo,
                    fa=phi_lo,
                    fpa=derphi_lo,
                    b=a_hi,
                    fb=phi_hi,
                    c=a_rec,
                    fc=phi_rec,
                )

            def true_fn_interpolation2(
                a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi, dalpha
            ):
                return self._quadmin(
                    point_1=a_lo,
                    obj_1=phi_lo,
                    grad_1=derphi_lo,
                    point_2=a_hi,
                    obj_2=phi_hi,
                )

            def true_fn_interpolation3(
                a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi, dalpha
            ):
                return a_lo + 0.5 * dalpha

            def false_fn_nan(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec):
                return tf.Variable(np.nan, dtype=tf.float64)

            def false_fn_interpolation(
                a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi, dalpha
            ):
                return a_j

            def interpolation(
                i,
                a_lo,
                phi_lo,
                derphi_lo,
                a_hi,
                phi_hi,
                a_rec,
                phi_rec,
                a,
                b,
                dalpha,
            ):
                cchk = self.delta1 * dalpha
                qchk = self.delta2 * dalpha

                a_j = tf.cond(
                    cond_interpolation1(i),
                    lambda: true_fn_interpolation1(
                        a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                    ),
                    lambda: false_fn_nan(
                        a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                    ),
                )
                a_j = tf.cond(
                    cond_interpolation2(i, a_j, a, b, cchk),
                    lambda: true_fn_interpolation2(
                        a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi, dalpha
                    ),
                    lambda: false_fn_interpolation(
                        a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi, dalpha
                    ),
                )
                a_j = tf.cond(
                    cond_interpolation3(a_j, a, b, qchk),
                    lambda: true_fn_interpolation3(
                        a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi, dalpha
                    ),
                    lambda: false_fn_interpolation(
                        a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi, dalpha
                    ),
                )

                return a_j

            a_j = interpolation(
                i, a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec, a, b, dalpha
            )

            # Check new value of a_j
            with tf.control_dependencies([phi0, derphi0]):
                phi_aj = self._objective_call(
                    self.weights + tf.cast(a_j, tf.float64) * search_direction,
                    x,
                    y,
                )
            cond1_aj = phi_aj > phi0 + self.c1 * a_j * derphi0
            cond2_aj = phi_aj >= phi_lo

            def true_fn_aj1():
                return phi_hi, a_hi, a_j, phi_aj

            def false_fn_aj1():
                return phi_rec, a_rec, a_hi, phi_hi

            phi_rec, a_rec, a_hi, phi_hi = tf.cond(
                tf.logical_or(cond1_aj, cond2_aj),
                lambda: true_fn_aj1(),
                lambda: false_fn_aj1(),
            )

            derphi_aj = tf.tensordot(
                self._gradient_call(self.weights + a_j * search_direction, x, y),
                search_direction,
                1,
            )
            cond3_aj = tf.math.abs(derphi_aj) <= -self.c2 * derphi0
            cond4_aj = derphi_aj * (a_hi - a_lo) >= 0

            def true_fn_aj2():
                return phi_hi, a_hi, a_lo, phi_lo, a_lo, phi_aj, derphi_aj

            def false_fn_aj2():
                return tf.cond(
                    cond4_aj,
                    lambda: (phi_rec, a_rec, a_lo, phi_lo, a_lo, phi_aj, derphi_aj),
                    lambda: (phi_lo, a_lo, a_j, phi_aj, a_j, phi_lo, derphi_aj),
                )

            phi_rec, a_rec, a_hi, phi_hi, a_lo, phi_lo, derphi_lo = tf.cond(
                cond3_aj,
                true_fn_aj2,
                false_fn_aj2,
            )
            i += 1
            if i > maxiter:
                # Failed to find a conforming step size
                a_star = tf.constant(0.0, dtype=tf.float64)
                break
        return a_star

    """
    
    def _zoom(
        self,
        a_lo,
        a_hi,
        phi_lo,
        phi_hi,
        derphi_lo,
        phi0,
        derphi0,
        search_direction,
        x,
        y,
    ):
        # TODO: maxiter as argument
        maxiter = 20
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

            if i > 0:
                cchk = delta1 * dalpha
                a_j = self._cubicmin(
                    a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                )
            if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
                qchk = delta2 * dalpha
                a_j = self._quadmin(
                    point_1=a_lo,
                    obj_1=phi_lo,
                    grad_1=derphi_lo,
                    point_2=a_hi,
                    obj_2=phi_hi,
                )
                if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                    a_j = a_lo + 0.5 * dalpha

            # Check new value of a_j
            phi_aj = self._objective_call(
                self.weights + as_Kfloat(a_j) * search_direction,
                x,
                y,
            )
            if (phi_aj > phi0 + self.c1 * a_j * derphi0) or (phi_aj >= phi_lo):
                phi_rec = phi_hi
                a_rec = a_hi
                a_hi = a_j
                phi_hi = phi_aj
            else:
                derphi_aj = tf.tensordot(
                    self._gradient_call(self.weights + a_j * search_direction, x, y),
                    search_direction,
                    1,
                )
                if tf.math.abs(derphi_aj) <= -self.c2 * derphi0:
                    a_star = a_j
                    break
                if derphi_aj * (a_hi - a_lo) >= 0:
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
            if i > maxiter:
                # Failed to find a conforming step size
                a_star = 0.0
                break
        return a_star
    
    """

    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        with tf.control_dependencies(
            [
                tf.debugging.assert_all_finite(
                    [a, fa, fpa, b, fb, c, fc], "Input values must be finite."
                )
            ]
        ):
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)

            d1 = tf.Variable(
                [[dc**2, -(db**2)], [-(dc**3), db**3]], dtype=tf.float64
            )
            d2 = tf.Variable([[fb - fa - C * db], [fc - fa - C * dc]], dtype=tf.float64)
            A = d1[0, 0] * d2[0] + d1[0, 1] * d2[1]
            B = d1[1, 0] * d2[0] + d1[1, 1] * d2[1]

            A = A / denom
            B = B / denom

            radical = B * B - 3 * A * C
            xmin = a + (-B + tf.sqrt(radical)) / (3 * A)

            return tf.where(
                tf.math.is_finite(xmin), xmin, tf.constant(np.nan, dtype=tf.float64)
            )

    """
    def _cubicmin(self, a, fa, fpa, b, fb, c, fc):
        # f(x) = A *(x-a)^3 + B*(x-a)^2 + C*(x-a) + D
        try:
            C = fpa
            db = b - a
            dc = c - a
            denom = (db * dc) ** 2 * (db - dc)
            d1 = np.empty((2, 2))
            d1[0, 0] = dc**2
            d1[0, 1] = -(db**2)
            d1[1, 0] = -(dc**3)
            d1[1, 1] = db**3
            # TODO: Hardcoded dtype
            [A, B] = np.dot(
                d1,
                np.asarray(
                    [fb - fa - C * db, fc - fa - C * dc], dtype="float64"
                ).flatten(),
            )
            A /= denom
            B /= denom
            radical = B * B - 3 * A * C
            xmin = a + (-B + np.sqrt(radical)) / (3 * A)
        except ArithmeticError:
            return None
        if not np.isfinite(xmin):
            return None
        return xmin
    """

    def _quadmin(self, point_1, obj_1, grad_1, point_2, obj_2):
        with tf.control_dependencies(
            [
                tf.debugging.assert_all_finite(
                    [point_1, obj_1, grad_1, point_2, obj_2],
                    "Input values must be finite.",
                )
            ]
        ):
            D = obj_1
            C = grad_1
            db = point_2 - point_1 * tf.constant(1.0, dtype=tf.float64)
            B = tf.where(
                tf.math.is_finite(db),
                (obj_2 - D - C * db) / (db * db),
                tf.constant(np.nan, dtype=tf.float64),
            )
            xmin = point_1 - C / (tf.constant(2.0, dtype=tf.float64) * B)

            return tf.where(
                tf.math.is_finite(xmin), xmin, tf.constant(np.nan, dtype=tf.float64)
            )

    """
    def _quadmin(self, point_1, obj_1, grad_1, point_2, obj_2):

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
    
    """

    def get_config(self):
        pass
