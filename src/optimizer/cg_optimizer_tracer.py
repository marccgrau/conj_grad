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
        self.a_j = tf.Variable(1.0, name="alpha_j", dtype=tf.float64)
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
        self.delta1 = tf.constant(
            0.2, name="delta1", dtype=tf.float64
        )  # cubic interpolant check
        self.delta2 = tf.constant(
            0.1, name="delta2", dtype=tf.float64
        )  # quadratic interpolant check
        self.xmin = tf.Variable(0.0, name="xmin", dtype=tf.float64)
        self.A = tf.Variable(0.0, name="A", dtype=tf.float64)
        self.B = tf.Variable(0.0, name="B", dtype=tf.float64)
        self.C = tf.Variable(0.0, name="C", dtype=tf.float64)
        self.D = tf.Variable(0.0, name="D", dtype=tf.float64)
        self.db = tf.Variable(0.0, name="db", dtype=tf.float64)
        self.dc = tf.Variable(0.0, name="dc", dtype=tf.float64)
        self.denom = tf.Variable(0.0, name="denom", dtype=tf.float64)
        self.radical = tf.Variable(0.0, name="radical", dtype=tf.float64)
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

        # conj grad vars
        self.d = tf.Variable(
            tf.zeros(shape=(param_count,), dtype=tf.float64),
            name="search_direction",
            dtype=tf.float64,
        )
        self.d_new = tf.Variable(
            tf.zeros(shape=(param_count,), dtype=tf.float64),
            name="search_direction",
            dtype=tf.float64,
        )
        self.r = tf.Variable(
            tf.zeros(shape=(param_count,), dtype=tf.float64),
            name="neg_grad",
            dtype=tf.float64,
        )
        self.r_new = tf.Variable(
            tf.zeros(shape=(param_count,), dtype=tf.float64),
            name="new_neg_grad",
            dtype=tf.float64,
        )
        self.grad = tf.Variable(
            tf.zeros(shape=(param_count,), dtype=tf.float64),
            name="grad",
            dtype=tf.float64,
        )
        self.obj_val = tf.Variable(0, name="loss_value", dtype=tf.float64)

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
        self.obj_val.assign(self.loss(y, self.model(x, training=True)))

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
        self.grad.assign(self._from_matrices_to_vector(grads))

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
        self.obj_val.assign(loss_value)
        self.grad.assign(self._from_matrices_to_vector(grads))

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

    def wolfe_and_conj_grad_step(self, x, y):
        self.wolfe_line_search(x=x, y=y)
        tf.print("alpha: ", self.alpha)
        self.conj_grad_step(x=x, y=y)
        self.j.assign_add(1)

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
        self._obj_func_and_grad_call(self.weights, x, y)
        self.r.assign(-self.grad)
        self.d.assign(self.r)

        def while_cond_update_step(iterate):
            return tf.math.logical_and(
                tf.math.less(self.j, self.max_iters),
                tf.math.equal(self._update_step_break, self.false_variable),
            )

        def body_update_step(iterate):
            self.wolfe_and_conj_grad_step(x=x, y=y)
            iterate = self.j
            return iterate

        iterate = np.float64(0)

        tf.while_loop(
            cond=while_cond_update_step,
            body=body_update_step,
            loop_vars=[iterate],
        )

    @tf.function
    def conj_grad_step(self, x, y):
        def alpha_zero_cond():
            return tf.math.equal(self.alpha, self.zero_variable)

        def true_fn():
            tf.print("Alpha is zero. Making no step.")

        def false_fn():
            self._save_new_model_weights(
                tf.math.add(self.weights, tf.math.multiply(self.alpha, self.d))
            )

            self._obj_func_and_grad_call(self.weights, x, y)
            # set r_{k+1}
            self.r_new.assign(tf.math.negative(self.grad))
            # Calculate Polak-Ribiére beta
            # PRP+ with max{beta{PR}, 0}
            self.beta.assign(
                tf.math.maximum(
                    tf.math.divide(
                        tf.reduce_sum(
                            tf.multiply(
                                self.r_new, tf.math.subtract(self.r_new, self.r)
                            )
                        ),
                        tf.reduce_sum(tf.multiply(self.r, self.r)),
                    ),
                    0,
                )
            )

            tf.print(f"beta: {self.beta}")
            # Determine new search direction for next iteration step
            self.d_new.assign(
                tf.math.add(self.r_new, tf.math.multiply(self.beta, self.d))
            )
            self.d.assign(self.d_new)
            self.r.assign(self.r_new)
            # TODO: Add convergence checks again!

        return tf.cond(
            alpha_zero_cond(),
            lambda: true_fn(),
            lambda: false_fn(),
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
    def wolfe_line_search(self, x=None, y=None):
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
        self._obj_func_and_grad_call(self.weights, x, y)
        self.phi0.assign(self.obj_val)
        # We need the directional derivative at 0 following the Wolfe Conditions
        # Thus, we get the gradient at 0 and multiply it with the search direction
        self.derphi0.assign(tf.tensordot(self.grad, self.d, 1))

        # Set alpha bounds
        # alpha0 = tf.Variable(0.0, dtype='float64')
        # alpha1 = tf.Variable(1.0, dtype='float64')
        # Optional setting of an alpha max, if defined
        self.alpha0.assign(self.zero_variable)
        self.alpha1.assign(tf.math.minimum(self.unity_variable, self.amax))

        # get objective value at a new possible position, i.e. w_k + alpha1 * d_k
        self._objective_call(
            tf.math.add(self.weights, tf.math.multiply(self.alpha1, self.d)), x, y
        )
        self.phi_a1.assign(self.obj_val)

        # Initialization of all variables for loop
        # Initial alpha_lo equivalent to alpha = 0
        self.phi_a0.assign(self.phi0)
        self.derphi_a0.assign(self.derphi0)
        self.i.assign(0)

        # While cond
        def while_cond(iterate):
            return tf.math.logical_and(
                tf.math.less(self.i, self.max_iters),
                tf.math.equal(self._break, self.false_variable),
            )

        # Define loop body
        def body(iterate):
            def init_cond():
                return tf.math.logical_or(
                    tf.math.equal(self.alpha1, self.zero_variable),
                    tf.math.equal(self.alpha0, self.amax),
                )

            # necessary for second check
            self._gradient_call(
                tf.math.add(self.weights, tf.math.multiply(self.alpha1, self.d)), x, y
            ),
            self.derphi_a1.assign(
                tf.tensordot(
                    self.grad,
                    self.d,
                    1,
                )
            )

            def first_cond():
                return tf.math.logical_or(
                    tf.math.greater(
                        self.phi_a1,
                        tf.math.add(
                            self.phi0,
                            tf.math.multiply(
                                tf.math.multiply(self.c1, self.alpha1), self.derphi0
                            ),
                        ),
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
                    self.d,
                    x,
                    y,
                )
                self.alpha.assign(self.alpha_star)
                self.i.assign_add(1)
                self._break.assign(self.true_variable)

            def second_cond():
                return tf.math.less_equal(
                    tf.math.abs(self.derphi_a1),
                    tf.math.multiply(-self.c2, self.derphi0),
                )

            def second_check_action():
                self.alpha_star.assign(self.alpha1)
                self._objective_call(
                    tf.math.add(
                        self.weights, tf.math.multiply(self.alpha_star, self.d)
                    ),
                    x,
                    y,
                )
                self.phi_star.assign(self.obj_val)
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
                    self.d,
                    x,
                    y,
                )
                self.alpha.assign(self.alpha_star)
                self.i.assign_add(1)
                self._break.assign(self.true_variable)

            def false_action():
                self._break.assign(self.false_variable)

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

            self.alpha2.assign(
                tf.math.multiply(tf.constant(2, dtype=tf.float64), self.alpha1)
            )

            self.alpha2.assign(tf.math.minimum(self.alpha2, self.amax))

            self.alpha0.assign(self.alpha1)
            self.alpha1.assign(self.alpha2)
            self.phi_a0.assign(self.phi_a1)
            self._objective_call(
                tf.math.add(self.weights, tf.math.multiply(self.alpha1, self.d)), x, y
            )
            self.phi_a1.assign(self.obj_val)
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
            self.dalpha.assign(tf.math.subtract(a_hi, a_lo))

            def cond_trial_step():
                return tf.math.less(self.dalpha, self.zero_variable)

            def true_fn_trial_step(a_hi, a_lo):
                return a_hi, a_lo

            def false_fn_trial_step(a_hi, a_lo):
                return a_lo, a_hi

            def trial_step_length(a_hi, a_lo):
                a, b = tf.cond(
                    cond_trial_step(),
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

            def cond_interpolation2(a, b, cchk):
                return tf.logical_or(
                    tf.equal(self.k, self.zero_variable),
                    tf.logical_or(
                        tf.equal(self.a_j, self.none_variable),
                        tf.logical_or(
                            tf.greater(self.a_j, tf.math.subtract(b, cchk)),
                            tf.less(self.a_j, tf.math.add(a, cchk)),
                        ),
                    ),
                )

            def cond_interpolation3(a, b, qchk):
                return tf.logical_or(
                    tf.equal(self.a_j, self.none_variable),
                    tf.logical_or(
                        tf.greater(self.a_j, tf.math.subtract(b, qchk)),
                        tf.less(self.a_j, tf.math.add(a, qchk)),
                    ),
                )

            def true_fn_interpolation1(
                a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
            ):
                self._cubicmin(
                    a=a_lo,
                    fa=phi_lo,
                    fpa=derphi_lo,
                    b=a_hi,
                    fb=phi_hi,
                    c=a_rec,
                    fc=phi_rec,
                )

            def true_fn_interpolation2(a_lo, phi_lo, derphi_lo, a_hi, phi_hi):
                self._quadmin(
                    point_1=a_lo,
                    obj_1=phi_lo,
                    grad_1=derphi_lo,
                    point_2=a_hi,
                    obj_2=phi_hi,
                )

            def true_fn_interpolation3(a_lo, phi_lo, derphi_lo, a_hi, phi_hi):
                self.a_j.assign(
                    tf.math.add(
                        a_lo,
                        tf.math.multiply(
                            tf.constant(0.5, dtype=tf.float64), self.dalpha
                        ),
                    )
                )

            def false_fn_nan(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec):
                self.a_j.assign(self.none_variable)

            def false_fn_interpolation(a_lo, phi_lo, derphi_lo, a_hi, phi_hi):
                self.a_j.assign(self.none_variable)

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
                cchk = tf.math.multiply(self.delta1, self.dalpha)
                qchk = tf.math.multiply(self.delta2, self.dalpha)

                tf.cond(
                    cond_interpolation1(),
                    lambda: true_fn_interpolation1(
                        a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                    ),
                    lambda: false_fn_nan(
                        a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec
                    ),
                )
                tf.cond(
                    cond_interpolation2(a, b, cchk),
                    lambda: true_fn_interpolation2(
                        a_lo, phi_lo, derphi_lo, a_hi, phi_hi
                    ),
                    lambda: false_fn_interpolation(
                        a_lo, phi_lo, derphi_lo, a_hi, phi_hi
                    ),
                )
                tf.cond(
                    cond_interpolation3(a, b, qchk),
                    lambda: true_fn_interpolation3(
                        a_lo, phi_lo, derphi_lo, a_hi, phi_hi
                    ),
                    lambda: false_fn_interpolation(
                        a_lo, phi_lo, derphi_lo, a_hi, phi_hi
                    ),
                )

            interpolation(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec, a, b)

            # Check new value of a_j
            self._objective_call(
                tf.math.add(self.weights, tf.math.multiply(self.a_j, self.d)),
                x,
                y,
            )
            phi_aj = self.obj_val

            def test_aj_cond(a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec):
                cond1_aj = tf.greater(
                    phi_aj,
                    tf.math.add(
                        phi0,
                        tf.math.multiply(tf.math.multiply(self.c1, self.a_j), derphi0),
                    ),
                )
                cond2_aj = tf.greater_equal(phi_aj, phi_lo)
                return tf.math.logical_or(cond1_aj, cond2_aj)

            def true_fn_aj1(a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec):
                return phi_hi, a_hi, a_j, phi_aj

            def false_fn_aj1(a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec):
                return phi_rec, a_rec, a_hi, phi_hi

            phi_rec, a_rec, a_hi, phi_hi = tf.cond(
                test_aj_cond(
                    self.a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec
                ),
                lambda: true_fn_aj1(
                    self.a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec
                ),
                lambda: false_fn_aj1(
                    self.a_j, phi_aj, phi0, derphi0, phi_lo, a_hi, phi_hi, a_rec
                ),
            )

            self._gradient_call(
                tf.math.add(self.weights, tf.math.multiply(self.a_j, self.d)), x, y
            )
            derphi_aj = tf.tensordot(
                self.grad,
                self.d,
                1,
            )

            def cond3_aj(derphi_aj, derphi0):
                return tf.math.less_equal(
                    tf.math.abs(derphi_aj),
                    tf.math.multiply(tf.math.negative(self.c2), derphi0),
                )

            def cond4_aj(derphi_aj, a_hi, a_lo, phi_hi, phi_lo):
                return tf.math.greater_equal(
                    tf.math.multiply(derphi_aj, tf.math.subtract(a_hi, a_lo)),
                    self.zero_variable,
                )

            tf.cond(
                cond3_aj(derphi_aj, derphi0),
                lambda: (
                    self.alpha_star.assign(self.a_j),
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

            a_lo = tf.cast(self.a_j, dtype=tf.float64)
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
        self.C.assign(fpa)
        self.db.assign(tf.math.subtract(b, a))
        self.dc.assign(tf.math.subtract(c, a))
        self.denom.assign(
            tf.math.multiply(
                tf.math.pow(tf.math.multiply(self.db, self.dc), 2),
                tf.math.subtract(self.db, self.dc),
            )
        )

        self.d1.assign(
            tf.reshape(
                [
                    [
                        tf.math.pow(self.dc, 2),
                        tf.math.negative(tf.math.pow(self.db, 2)),
                    ],
                    [
                        tf.math.negative(tf.math.pow(self.dc, 3)),
                        tf.math.pow(self.db, 3),
                    ],
                ],
                [2, 2],
            )
        )
        self.d2.assign(
            tf.reshape(
                [
                    [
                        tf.math.subtract(
                            tf.math.subtract(fb, fa), tf.math.multiply(self.C, self.db)
                        )
                    ],
                    [
                        tf.math.subtract(
                            tf.math.subtract(fc, fa), tf.math.multiply(self.C, self.dc)
                        )
                    ],
                ],
                [2, 1],
            )
        )
        self.A.assign(
            tf.squeeze(
                tf.math.divide(
                    tf.math.add(
                        tf.math.multiply(self.d1[0, 0], self.d2[0]),
                        tf.math.multiply(self.d1[0, 1], self.d2[1]),
                    ),
                    self.denom,
                )
            )
        )
        self.B.assign(
            tf.squeeze(
                tf.math.divide(
                    tf.math.add(
                        tf.math.multiply(self.d1[1, 0], self.d2[0]),
                        tf.math.multiply(self.d1[1, 1], self.d2[1]),
                    ),
                    self.denom,
                )
            )
        )

        radical = tf.math.subtract(
            tf.math.multiply(self.B, self.B),
            tf.math.multiply(
                tf.math.multiply(tf.constant(3, dtype=tf.float64), self.A), self.C
            ),
        )
        self.xmin.assign(
            tf.math.add(
                a,
                tf.math.divide(
                    tf.math.add(tf.math.negative(self.B), tf.sqrt(radical)),
                    tf.math.multiply(tf.constant(3, dtype=tf.float64), self.A),
                ),
            )
        )

        self.a_j.assign(
            tf.where(tf.math.is_finite(self.xmin), self.xmin, self.none_variable)
        )

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
        self.D.assign(obj_1)
        self.C.assign(grad_1)
        self.db.assign(
            tf.math.subtract(
                point_2, tf.math.multiply(point_1, tf.constant(1.0, dtype=tf.float64))
            )
        )
        self.B.assign(
            tf.where(
                tf.math.is_finite(self.db),
                tf.math.divide(
                    (
                        tf.math.subtract(
                            obj_2,
                            tf.math.subtract(self.D, tf.math.multiply(self.C, self.db)),
                        )
                    ),
                    tf.math.multiply(self.db, self.db),
                ),
                self.none_variable,
            )
        )
        self.xmin.assign(
            tf.math.subtract(
                point_1,
                tf.math.divide(
                    self.C, tf.math.multiply(tf.constant(2.0, dtype=tf.float64), self.B)
                ),
            )
        )

        self.a_j.assign(
            tf.where(tf.math.is_finite(self.xmin), self.xmin, self.none_variable)
        )

    def get_config(self):
        pass
