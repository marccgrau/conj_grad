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
        self.max_iter_zoom = tf.Variable(10, name="max_iter_zoom", dtype=tf.float64)
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
        self.beta = tf.Variable(0.0, name="beta", dtype=tf.float64)
        self.alpha_star = tf.Variable(0.0, name="alpha_star", dtype=tf.float64)
        self.alpha0 = tf.Variable(0.0, name="alpha0", dtype=tf.float64)
        self.alpha1 = tf.Variable(1.0, name="alpha1", dtype=tf.float64)
        self.alpha2 = tf.Variable(1.0, name="alpha2", dtype=tf.float64)
        self.dalpha = tf.Variable(0.0, name="dalpha", dtype=tf.float64)
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
        self._zoom_break = tf.Variable(False, dtype=bool)
        # zoom function
        self.delta1 = 0.2  # cubic interpolant check
        self.delta2 = 0.1  # quadratic interpolant check
        # counters
        self.j = tf.Variable(0, name="update_step_counter", dtype=tf.float64)
        self.i = tf.Variable(0, name="wolfe_ls_counter", dtype=tf.float64)
        self.k = tf.Variable(0, name="zoom_counter", dtype=tf.float64)
        # constants
        self.zero_variable = tf.Variable(0, dtype=tf.float64)
        self.unity_variable = tf.Variable(1, dtype=tf.float64)
        self.false_variable = tf.Variable(False, dtype=bool)
        self.true_variable = tf.Variable(True, dtype=bool)
        self.none_variable = tf.Variable(np.nan, dtype=tf.float64)
        # empty variables for interpolation
        self.d1 = tf.Variable([[0, 0], [0, 0]], dtype=tf.float64)
        self.d2 = tf.Variable([[0], [0]], dtype=tf.float64)

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

    def wolfe_and_conj_grad_step(self, maxiter, d, r, obj_val, x, y):
        self.wolfe_line_search(maxiter=maxiter, search_direction=d, x=x, y=y)
        tf.print("alpha: ", self.alpha)
        d, r, obj_val = self.conj_grad_step(d, r, obj_val, x, y)
        self.j.assign_add(1)
        return d, r, obj_val

    @tf.function
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
        self.j.assign(0)

        def while_cond_update_step(d, r, obj_val):
            return tf.math.logical_and(
                tf.math.less(self.j, self.max_iters),
                tf.math.equal(self._update_step_break, tf.constant(False)),
            )

        body_update_step = lambda d, r, obj_val: self.wolfe_and_conj_grad_step(
            maxiter=10, d=d, r=r, obj_val=obj_val, x=x, y=y
        )

        obj_val, grad = self._obj_func_and_grad_call(self.weights, x, y)
        r = -grad
        d = r

        d, r, obj_val = tf.while_loop(
            cond=while_cond_update_step,
            body=body_update_step,
            loop_vars=[d, r, obj_val],
        )
        # for i in range(10):
        # Perform line search to determine alpha_star
        # self.wolfe_line_search(maxiter=10, search_direction=d, x=x, y=y)
        # logger.info(f"alpha after line search: {self.alpha}")
        # d, r, obj_val = self.conj_grad_step(self.alpha, d, r, obj_val, x, y)
        # if self.alpha == 0:
        #    break

    @tf.function
    def conj_grad_step(self, d, r, obj_val, x, y):
        def alpha_zero_cond():
            return tf.math.equal(self.alpha, self.zero_variable)

        def true_fn(d, r, obj_val):
            tf.print("Alpha is zero. Making no step.")
            return d, r, obj_val

        def false_fn(d, r, obj_val):
            w_new = self.weights + self.alpha * d
            self._save_new_model_weights(w_new)

            obj_val_new, grad_new = self._obj_func_and_grad_call(self.weights, x, y)
            # set r_{k+1}
            r_new = -grad_new
            # Calculate Polak-Ribiére beta
            # PRP+ with max{beta{PR}, 0}
            self.beta.assign(
                tf.math.maximum(
                    tf.reduce_sum(tf.multiply(r_new, r_new - r))
                    / tf.reduce_sum(tf.multiply(r, r)),
                    0,
                )
            )

            tf.print(f"beta: {self.beta}")
            # Determine new search direction for next iteration step
            d_new = r_new + self.beta * d
            # TODO: Add convergence checks again!
            return d_new, r_new, obj_val_new

        return tf.cond(
            alpha_zero_cond(),
            lambda: true_fn(d, r, obj_val),
            lambda: false_fn(d, r, obj_val),
        )

        """
        # Check for convergence
        if tf.reduce_sum(tf.abs(obj_val_new - obj_val)) < self.tol:
            tf.print(f"Stop NLCG, no sufficient decrease in value function")
            return d_new, r_new, obj_val_new
        # check if vector norm is smaller than the tolerance
        if tf.norm(r_new) <= self.tol:
            tf.print(f"Stop NLCG, gradient norm smaller than tolerance")
            return d_new, r_new, obj_val_new
        return d_new, r_new, obj_val_new
        """

    @tf.function
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
        return self.model.trainable_variables

    @tf.function
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
        self.alpha1.assign(tf.math.minimum(self.alpha1, self.amax))

        # get objective value at a new possible position, i.e. w_k + alpha1 * d_k
        self.phi_a1.assign(
            self._objective_call(self.weights + self.alpha1 * search_direction, x, y)
        )

        # Initialization of all variables for loop
        # Initial alpha_lo equivalent to alpha = 0
        self.phi_a0.assign(self.phi0)
        self.derphi_a0.assign(self.derphi0)
        self.i.assign(0)

        # While cond
        def while_cond(iterate):
            return tf.math.logical_and(
                tf.math.less(self.i, self.max_iters),
                tf.math.equal(self._break, tf.constant(False)),
            )

        # Define loop body
        def body(iterate):
            def init_cond():
                return tf.math.logical_or(
                    tf.math.equal(self.alpha1, self.zero_variable),
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
                        tf.math.greater(self.i, self.unity_variable),
                    ),
                )

            def first_check_action():
                # _zoom will assign alpha_star
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
                self.alpha.assign(self.alpha_star)
                self.i.assign_add(1)
                self._break.assign(self.true_variable)

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
                self.i.assign_add(1)
                self._break.assign(self.true_variable)

            def third_cond():
                return tf.math.greater_equal(self.derphi_a1, self.zero_variable)

            def third_check_action():
                # _zoom will assign alpha star
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
                self.alpha.assign(self.alpha_star)
                self.i.assign_add(1)
                self._break.assign(self.true_variable)

            def false_action():
                self._break.assign(tf.constant(False))

            def final_cond():
                return tf.math.equal(self.i, self.max_iters)

            tf.cond(
                init_cond(),
                lambda: (
                    self.alpha_star.assign(self.zero_variable),
                    self.i.assign_add(1),
                    self._break.assign(self.true_variable),
                ),
                lambda: (
                    self.alpha_star.assign(self.zero_variable),
                    self.i.assign_add(0),
                    self._break.assign(self.false_variable),
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

            self.i.assign_add(1)
            # if no break occurs, then we have not found a suitable alpha after maxiter
            tf.cond(
                final_cond(),
                lambda: (
                    self.alpha_star.assign(self.alpha1),
                    self.alpha.assign(self.alpha_star),
                    self._break.assign(self.true_variable),
                ),
                lambda: (
                    self.alpha_star,
                    self.alpha,
                    self._break.assign(self.false_variable),
                ),
            )
            iterate = self.i
            return iterate

        iterate = np.float64(0)
        # While loop
        tf.while_loop(while_cond, body, [iterate])

    @tf.function
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
        phi_rec = phi0
        a_rec = self.zero_variable

        def zoom_while_cond(
            a_lo, a_hi, a_rec, phi_lo, phi_hi, phi_rec, derphi_lo, phi0, derphi0
        ):
            return tf.math.logical_and(
                tf.less(self.k, self.max_iter_zoom),
                tf.math.equal(self._zoom_break, self.false_variable),
            )

        def zoom_body(
            a_lo, a_hi, a_rec, phi_lo, phi_hi, phi_rec, derphi_lo, phi0, derphi0
        ):
            self.dalpha.assign(a_hi - a_lo)
            dalpha_cond = tf.math.less(self.dalpha, self.zero_variable)
            self.dalpha.assign(tf.where(dalpha_cond, -self.dalpha, self.dalpha))

            def cond_trial_step(a_hi, a_lo):
                return tf.less(self.dalpha, 0)

            def true_fn_trial_step(a_hi, a_lo):
                return a_hi, a_lo

            def false_fn_trial_step(a_hi, a_lo):
                return a_lo, a_hi

            def trial_step_length(a_hi, a_lo):
                a, b = tf.cond(
                    cond_trial_step(a_hi, a_lo),
                    lambda: true_fn_trial_step(a_hi, a_lo),
                    lambda: false_fn_trial_step(a_hi, a_lo),
                )
                return a, b

            a, b = trial_step_length(a_hi, a_lo)

            # minimizer of cubic interpolant
            # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
            #
            # if the result is too close to the end points (or out of the
            # interval), then use quadratic interpolation with phi_lo,
            # derphi_lo and phi_hi if the result is still too close to the
            # end points (or out of the interval) then use bisection

            def cond_interpolation1():
                return tf.greater(self.k, self.zero_variable)

            def cond_interpolation2(a_j, a, b, cchk):
                return tf.logical_or(
                    tf.equal(self.k, 0),
                    tf.logical_or(
                        tf.equal(a_j, self.none_variable),
                        tf.logical_or(
                            tf.greater(a_j, b - cchk), tf.less(a_j, a + cchk)
                        ),
                    ),
                )

            def cond_interpolation3(a_j, a, b, qchk):
                return tf.logical_or(
                    tf.equal(a_j, self.none_variable),
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

            def true_fn_interpolation2(a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi):
                return self._quadmin(
                    point_1=a_lo,
                    obj_1=phi_lo,
                    grad_1=derphi_lo,
                    point_2=a_hi,
                    obj_2=phi_hi,
                )

            def true_fn_interpolation3(a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi):
                return a_lo + 0.5 * self.dalpha

            def false_fn_nan(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec):
                return self.none_variable

            def false_fn_interpolation(a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi):
                return self.none_variable

            def interpolation(
                a_lo,
                phi_lo,
                derphi_lo,
                a_hi,
                phi_hi,
                a_rec,
                phi_rec,
                a,
                b,
            ):
                cchk = self.delta1 * self.dalpha
                qchk = self.delta2 * self.dalpha

                a_j = tf.cond(
                    cond_interpolation1(),
                    lambda: true_fn_interpolation1(
                        a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                    ),
                    lambda: false_fn_nan(
                        a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                    ),
                )
                a_j = tf.cond(
                    cond_interpolation2(a_j, a, b, cchk),
                    lambda: true_fn_interpolation2(
                        a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi
                    ),
                    lambda: false_fn_interpolation(
                        a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi
                    ),
                )
                a_j = tf.cond(
                    cond_interpolation3(a_j, a, b, qchk),
                    lambda: true_fn_interpolation3(
                        a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi
                    ),
                    lambda: false_fn_interpolation(
                        a_j, a_lo, phi_lo, derphi_lo, a_hi, phi_hi
                    ),
                )
                return a_j

            a_j = interpolation(
                a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec, a, b
            )

            # Check new value of a_j
            phi_aj = self._objective_call(
                self.weights + a_j * search_direction,
                x,
                y,
            )

            def test_aj_cond(a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec):
                cond1_aj = tf.greater(phi_aj, phi0 + self.c1 * a_j * derphi0)
                cond2_aj = tf.greater_equal(phi_aj, phi_lo)
                return tf.math.logical_or(cond1_aj, cond2_aj)

            def true_fn_aj1(a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec):
                return phi_hi, a_hi, a_j, phi_aj

            def false_fn_aj1(a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec):
                return phi_rec, a_rec, a_hi, phi_hi

            phi_rec, a_rec, a_hi, phi_hi = tf.cond(
                test_aj_cond(a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec),
                lambda: true_fn_aj1(
                    a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec
                ),
                lambda: false_fn_aj1(
                    a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec
                ),
            )

            derphi_aj = tf.tensordot(
                self._gradient_call(self.weights + a_j * search_direction, x, y),
                search_direction,
                1,
            )

            def cond3_aj(derphi_aj, derphi0):
                return tf.math.less_equal(tf.math.abs(derphi_aj), -self.c2 * derphi0)

            def cond4_aj(derphi_aj, a_hi, a_lo, phi_hi, phi_lo):
                return tf.math.greater_equal(
                    derphi_aj * (a_hi - a_lo), self.zero_variable
                )

            tf.cond(
                cond3_aj(derphi_aj, derphi0),
                lambda: (
                    self.alpha_star.assign(a_j),
                    self._zoom_break.assign(self.true_variable),
                ),
                lambda: (self.alpha_star, self._zoom_break.assign(self.false_variable)),
            )

            def true_fn_aj4(derphi_aj, a_hi, a_lo, phi_hi, phi_lo):
                return phi_hi, a_hi, a_lo, phi_lo

            def false_fn_aj4(derphi_aj, a_hi, a_lo, phi_hi, phi_lo):
                return phi_lo, a_lo, a_lo, phi_lo

            phi_rec, a_rec, a_hi, phi_hi = tf.cond(
                cond4_aj(derphi_aj, a_hi, a_lo, phi_hi, phi_lo),
                lambda: true_fn_aj4(derphi_aj, a_hi, a_lo, phi_hi, phi_lo),
                lambda: false_fn_aj4(derphi_aj, a_hi, a_lo, phi_hi, phi_lo),
            )

            a_lo = tf.cast(a_j, dtype=tf.float64)
            phi_lo = phi_aj
            derphi_lo = derphi_aj
            self.k.assign_add(1)

            def final_cond():
                return tf.less(self.k, self.max_iter_zoom)

            tf.cond(
                final_cond(),
                lambda: (
                    self.alpha_star.assign(self.zero_variable),
                    self._zoom_break.assign(self.true_variable),
                ),
                lambda: (self.alpha_star, self._zoom_break.assign(self.false_variable)),
            )

            return a_lo, a_hi, a_rec, phi_lo, phi_hi, phi_rec, derphi_lo, phi0, derphi0

        tf.while_loop(
            zoom_while_cond,
            zoom_body,
            [a_lo, a_hi, a_rec, phi_lo, phi_hi, phi_rec, derphi_lo, phi0, derphi0],
        )

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
        """
        with tf.control_dependencies(
            [
                tf.debugging.assert_all_finite(
                    [a, fa, fpa, b, fb, c, fc], "Input values must be finite."
                )
            ]
        ):
        """
        C = fpa
        db = b - a
        dc = c - a
        denom = (db * dc) ** 2 * (db - dc)

        self.d1.assign(
            tf.reshape([[dc**2, -(db**2)], [-(dc**3), db**3]], [2, 2])
        )
        self.d2.assign(tf.reshape([[fb - fa - C * db], [fc - fa - C * dc]], [2, 1]))
        A = self.d1[0, 0] * self.d2[0] + self.d1[0, 1] * self.d2[1]
        B = self.d1[1, 0] * self.d2[0] + self.d1[1, 1] * self.d2[1]

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
        """
        with tf.control_dependencies(
            [
                tf.debugging.assert_all_finite(
                    [point_1, obj_1, grad_1, point_2, obj_2],
                    "Input values must be finite.",
                )
            ]
        ):
        """
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
