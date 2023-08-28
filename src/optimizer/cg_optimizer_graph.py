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
        max_iters=2,
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
        self.max_iter_zoom = tf.Variable(5, name="max_iter_zoom", dtype=tf.float64)
        self.tol = tf.Variable(tol, name="tolerance")
        self.c1 = tf.Variable(c1, name="c1")
        self.c2 = tf.Variable(c2, name="c2")
        self.amax = tf.Variable(amax, name="amax")
        self.alpha = tf.Variable(initial_value=0.0, name="alpha", dtype=tf.float64)
        self.model = model
        self.loss = loss
        # function call counters
        self.objective_tracker = tf.Variable(0, name="objective_tracker", dtype=tf.float64)
        self.grad_tracker = tf.Variable(0, name="gradient_tracker", dtype=tf.float64)
        # model specifics
        self._weight_shapes = tf.shape_n(self.model.trainable_weights)
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
        self.a_lo = tf.Variable(0.0, name="a_lo", dtype=tf.float64)
        self.phi_lo = tf.Variable(0.0, name="phi_lo", dtype=tf.float64)
        self.derphi_lo = tf.Variable(0.0, name="derphi_lo", dtype=tf.float64)
        self.a_hi = tf.Variable(0.0, name="a_hi", dtype=tf.float64)
        self.phi_hi = tf.Variable(0.0, name="phi_hi", dtype=tf.float64)
        self.derphi_hi = tf.Variable(0.0, name="derphi_hi", dtype=tf.float64)
        self.phi_rec = tf.Variable(0.0, name="phi_rec", dtype=tf.float64)
        self.a_rec = tf.Variable(0.0, name="a_rec", dtype=tf.float64)
        self.a_point = tf.Variable(0.0, name="a_point", dtype=tf.float64)
        self.b_point = tf.Variable(0.0, name="b_point", dtype=tf.float64)
        self.c_point = tf.Variable(0.0, name="c_point", dtype=tf.float64)
        self.cchk = tf.Variable(0.0, name="cchk", dtype=tf.float64)
        self.qchk = tf.Variable(0.0, name="qchk", dtype=tf.float64)
        self.phi_aj = tf.Variable(0.0, name="phi_aj", dtype=tf.float64)
        self.derphi_aj = tf.Variable(0.0, name="derphi_aj", dtype=tf.float64)
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
        # loop vars
        self._update_step_iterate = tf.Variable(0, name="update_step_counter", dtype=tf.float64)
        self._wolfe_iterate = tf.Variable(0, name="wolfe_ls_counter", dtype=tf.float64)
        self._zoom_iterate = tf.Variable(0, name="zoom_counter", dtype=tf.float64)
        self._wolfe_break = tf.Variable(False, dtype=bool)
        self._update_step_break = tf.Variable(False, dtype=bool)
        self._zoom_break = tf.Variable(False, dtype=bool)
        # constants
        self.zero_variable = tf.Variable(0.0, dtype=tf.float64)
        self.unity_variable = tf.Variable(1, dtype=tf.float64)
        self.false_variable = tf.Variable(False, dtype=bool)
        self.true_variable = tf.Variable(True, dtype=bool)
        self.none_variable = tf.Variable(np.nan, dtype=tf.float64)
        # empty variables for interpolation
        self.d1 = tf.Variable([[0, 0], [0, 0]], dtype=tf.float64)
        self.d2 = tf.Variable([[0], [0]], dtype=tf.float64)
        
        self.iterate_wolfe = tf.Variable(0)
        self.iterate_update = tf.Variable(0)

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
            self._from_matrices_to_vector(model.trainable_weights), name="weights"
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
            self.model.trainable_weights[i].assign(param)



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
        self.objective_tracker.assign_add(self.unity_variable)
        self.obj_val.assign(self.loss(y, self.model(x, training=True)))


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
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.grad_tracker.assign_add(self.unity_variable)
        self.objective_tracker.assign_add(self.unity_variable)
        self.grad.assign(self._from_matrices_to_vector(grads))


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
        grads = tape.gradient(loss_value, self.model.trainable_weights)
        self.grad_tracker.assign_add(self.unity_variable)
        self.objective_tracker.assign_add(self.unity_variable)
        self.obj_val.assign(loss_value)
        self.grad.assign(self._from_matrices_to_vector(grads))

        
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
        # TODO: Set break condition for update step
        self.wolfe_line_search(x=x, y=y)
        # tf.print(self.alpha_star)
        self.conj_grad_step(x=x, y=y)
        self._update_step_iterate.assign_add(self.unity_variable)


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
        self._update_step_iterate.assign(self.zero_variable)
        self._update_step_break.assign(self.false_variable)
        
        self._obj_func_and_grad_call(self.weights, x, y)
        self.r.assign(tf.math.negative(self.grad))
        self.d.assign(self.r)
        self._update_step_break.assign(self.false_variable)

        def while_cond_update_step(iterate):
            return tf.math.logical_and(
                tf.math.less(self._update_step_iterate, self.max_iters),
                tf.math.equal(self._update_step_break, self.false_variable),
            )

        def body_update_step(iterate):
            self.wolfe_and_conj_grad_step(x=x, y=y)

            def check_alpha_zero():
                return tf.math.equal(self.alpha, self.zero_variable)

            def stop_update():
                self._update_step_break.assign(self.true_variable)

            def cont_update():
                pass

            tf.cond(
                check_alpha_zero(),
                stop_update,
                cont_update,
            )
            return (self._update_step_iterate,)

        tf.while_loop(
            cond=while_cond_update_step,
            body=body_update_step,
            loop_vars=[self._update_step_iterate],
        )


    def conj_grad_step(self, x, y):
        def alpha_zero_cond():
            return tf.math.equal(self.alpha, self.zero_variable)

        def true_fn():
            tf.print("Alpha is zero. Making no step.")

        def false_fn():
            tf.print(self.alpha)
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

            tf.print(self.beta)
            # Determine new search direction for next iteration step
            self.d_new.assign(
                tf.math.add(self.r_new, tf.math.multiply(self.beta, self.d))
            )
            self.d.assign(self.d_new)
            self.r.assign(self.r_new)
            # TODO: Add convergence checks again!

        tf.cond(
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
        self._save_new_model_weights(self._from_matrices_to_vector(vars))
        self.update_step(x, y)
        self._save_new_model_weights(self.weights)
        return self.model.trainable_weights

    
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
        self._wolfe_iterate.assign(self.zero_variable)
        self._wolfe_break.assign(self.false_variable)
        
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

        # While cond
        def while_cond(iterate):
            return tf.math.logical_and(
                tf.math.less(self._wolfe_iterate, self.max_iters),
                tf.math.equal(self._wolfe_break, self.false_variable),
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
                        tf.math.greater(self._wolfe_iterate, self.unity_variable),
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
                    x,
                    y,
                )
                self.alpha.assign(self.alpha_star)
                # self._wolfe_iterate.assign_add(1)
                self._wolfe_break.assign(self.true_variable)
                return self._wolfe_break

            def second_cond():
                return tf.math.less_equal(
                    tf.math.abs(self.derphi_a1),
                    tf.math.multiply(tf.math.negative(self.c2), self.derphi0),
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
                # self._wolfe_iterate.assign_add(1)
                self._wolfe_break.assign(self.true_variable)
                return self._wolfe_break

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
                    x,
                    y,
                )
                self.alpha.assign(self.alpha_star)
                # self._wolfe_iterate.assign_add(1)
                self._wolfe_break.assign(self.true_variable)
                return self._wolfe_break

            def false_action():
                self._wolfe_break.assign(self.false_variable)
                return self._wolfe_break

            tf.cond(
                init_cond(),
                lambda: (
                    self.alpha_star.assign(self.zero_variable),
                    # self._wolfe_iterate.assign_add(1),
                    self._wolfe_break.assign(self.true_variable),
                ),
                lambda: (
                    self.alpha_star.assign(self.zero_variable),
                    # self._wolfe_iterate.assign_add(0),
                    self._wolfe_break.assign(self.false_variable),
                ),
            )

            tf.cond(
                first_cond(),
                lambda: first_check_action(),
                lambda: tf.cond(
                    second_cond(),
                    lambda: second_check_action(),
                    lambda: tf.cond(
                        third_cond(),
                        lambda: third_check_action(),
                        lambda: false_action(),
                    ),
                ),
            )

            def solution_found_cond():
                return tf.math.equal(self._wolfe_break, self.true_variable)

            def solution_found():
                pass

            def solution_not_found():
                self.alpha2.assign(
                    tf.math.multiply(tf.constant(2, dtype=tf.float64), self.alpha1)
                )

                self.alpha2.assign(tf.math.minimum(self.alpha2, self.amax))

                self.alpha0.assign(self.alpha1)
                self.alpha1.assign(self.alpha2)
                self.phi_a0.assign(self.phi_a1)
                self._objective_call(
                    tf.math.add(self.weights, tf.math.multiply(self.alpha1, self.d)),
                    x,
                    y,
                )
                self.phi_a1.assign(self.obj_val)
                self.derphi_a0.assign(self.derphi_a1)

            tf.cond(
                solution_found_cond(),
                lambda: solution_found(),
                lambda: solution_not_found(),
            )

            self._wolfe_iterate.assign_add(self.unity_variable)
            return (self._wolfe_iterate,)
    
        
        # While loop
        tf.while_loop(while_cond, body, [self._wolfe_iterate])


    def _zoom(
        self,
        a_lo,
        a_hi,
        phi_lo,
        phi_hi,
        derphi_lo,
        x,
        y,
    ):
        """
        Zoom stage of approximate line search satisfying strong Wolfe conditions.
        """
        self._zoom_iterate.assign(self.zero_variable)
        self._zoom_break.assign(self.false_variable)
        
        self.phi_rec = self.phi0
        self.a_hi.assign(a_hi)
        self.a_lo.assign(a_lo)
        self.phi_lo.assign(phi_lo)
        self.phi_hi.assign(phi_hi)
        self.derphi_lo.assign(derphi_lo)
        self.a_rec.assign(self.zero_variable)
        self._zoom_break.assign(self.false_variable)
        self._zoom_iterate.assign(self.zero_variable)

        def zoom_while_cond(iterate):
            return tf.math.logical_and(
                tf.less(self._zoom_iterate, self.max_iter_zoom),
                tf.math.equal(self._zoom_break, self.false_variable),
            )

        def zoom_body(iterate):
            self.dalpha.assign(tf.math.subtract(self.a_hi, self.a_lo))

            def cond_trial_step():
                return tf.math.less(self.dalpha, self.zero_variable)

            def true_fn_trial_step():
                self.a_point.assign(self.a_hi)
                self.b_point.assign(self.a_lo)

            def false_fn_trial_step():
                self.a_point.assign(self.a_lo)
                self.b_point.assign(self.a_hi)

            tf.cond(
                cond_trial_step(),
                lambda: true_fn_trial_step(),
                lambda: false_fn_trial_step(),
            )

            # minimizer of cubic interpolant
            # (uses phi_lo, derphi_lo, phi_hi, and the most recent value of phi)
            #
            # if the result is too close to the end points (or out of the
            # interval), then use quadratic interpolation with phi_lo,
            # derphi_lo and phi_hi if the result is still too close to the
            # end points (or out of the interval) then use bisection

            def cond_interpolation1():
                return tf.greater(self._zoom_iterate, self.zero_variable)

            def cond_interpolation2():
                return tf.logical_or(
                    tf.equal(self._zoom_iterate, self.zero_variable),
                    tf.logical_or(
                        tf.equal(self.a_j, self.none_variable),
                        tf.logical_or(
                            tf.greater(
                                self.a_j, tf.math.subtract(self.b_point, self.cchk)
                            ),
                            tf.less(self.a_j, tf.math.add(self.a_point, self.cchk)),
                        ),
                    ),
                )

            def cond_interpolation3():
                return tf.logical_or(
                    tf.equal(self.a_j, self.none_variable),
                    tf.logical_or(
                        tf.greater(self.a_j, tf.math.subtract(self.b_point, self.qchk)),
                        tf.less(self.a_j, tf.math.add(self.a_point, self.qchk)),
                    ),
                )

            def true_fn_interpolation1():
                self._cubicmin()

            def true_fn_interpolation2():
                self._quadmin()
                tf.cond(
                    cond_interpolation3(),
                    lambda: true_fn_interpolation3(),
                    lambda: false_fn_interpolation(),
                )

            def true_fn_interpolation3():
                self.a_j.assign(
                    tf.math.add(
                        self.a_lo,
                        tf.math.multiply(
                            tf.constant(0.5, dtype=tf.float64), self.dalpha
                        ),
                    )
                )

            def false_fn_nan():
                self.a_j.assign(self.none_variable)

            def false_fn_interpolation():
                pass

            def interpolation():
                self.cchk.assign(tf.math.multiply(self.delta1, self.dalpha))
                self.qchk.assign(tf.math.multiply(self.delta2, self.dalpha))

                tf.cond(
                    cond_interpolation1(),
                    lambda: true_fn_interpolation1(),
                    lambda: false_fn_nan(),
                )
                tf.cond(
                    cond_interpolation2(),
                    lambda: true_fn_interpolation2(),
                    lambda: false_fn_interpolation(),
                )

            interpolation()

            # Check new value of a_j
            self._objective_call(
                tf.math.add(self.weights, tf.math.multiply(self.a_j, self.d)),
                x,
                y,
            )
            self.phi_aj.assign(self.obj_val)

            def test_aj_cond():
                return tf.logical_or(
                    tf.greater(
                        self.phi_aj,
                        tf.math.add(
                            self.phi0,
                            tf.math.multiply(
                                tf.math.multiply(self.c1, self.a_j), self.derphi0
                            ),
                        ),
                    ),
                    tf.greater_equal(self.phi_aj, self.phi_lo),
                )

            def true_fn_aj1():
                self.phi_rec.assign(self.phi_hi)
                self.a_rec.assign(self.a_hi)
                self.a_hi.assign(self.a_j)
                self.phi_hi.assign(self.phi_aj)

            def false_fn_aj1():
                self._gradient_call(
                    tf.math.add(self.weights, tf.math.multiply(self.a_j, self.d)), x, y
                )
                self.derphi_aj.assign(
                    tf.tensordot(
                        self.grad,
                        self.d,
                        1,
                    )
                )

                def cond3_aj():
                    return tf.math.less_equal(
                        tf.math.abs(self.derphi_aj),
                        tf.math.multiply(tf.math.negative(self.c2), self.derphi0),
                    )

                def true_fn_aj3():
                    self.alpha_star.assign(self.a_j)
                    self._zoom_break.assign(self.true_variable)

                def false_fn_aj3():
                    pass

                tf.cond(
                    cond3_aj(),
                    true_fn_aj3,
                    false_fn_aj3,
                )

                def cond4_aj():
                    return tf.math.greater_equal(
                        tf.math.multiply(
                            self.derphi_aj, tf.math.subtract(self.a_hi, self.a_lo)
                        ),
                        self.zero_variable,
                    )

                def true_fn_aj4():
                    self.phi_rec.assign(self.phi_hi)
                    self.a_rec.assign(self.a_hi)
                    self.a_hi.assign(self.a_lo)
                    self.phi_hi.assign(self.phi_lo)

                def false_fn_aj4():
                    self.phi_rec.assign(self.phi_lo)
                    self.a_rec.assign(self.a_lo)

                def stop_cond():
                    return tf.math.equal(self._zoom_break, self.true_variable)

                def pass_func():
                    pass

                def no_break_func():
                    tf.cond(cond4_aj(), true_fn_aj4, false_fn_aj4)
                    self.a_lo.assign(self.a_j)
                    self.phi_lo.assign(self.phi_aj)
                    self.derphi_lo.assign(self.derphi_aj)

                tf.cond(stop_cond(), pass_func, no_break_func)

            tf.cond(
                test_aj_cond(),
                true_fn_aj1,
                false_fn_aj1,
            )

            self._zoom_iterate.assign_add(self.unity_variable)
            return (self._zoom_iterate,)

        tf.while_loop(
            zoom_while_cond,
            zoom_body,
            [self._zoom_iterate],
        )


    def _cubicmin(self):
        self.C.assign(self.derphi_lo)
        self.db.assign(tf.math.subtract(self.a_hi, self.a_lo))
        self.dc.assign(tf.math.subtract(self.a_rec, self.a_lo))
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
                            tf.math.subtract(self.phi_hi, self.phi_lo),
                            tf.math.multiply(self.C, self.db),
                        )
                    ],
                    [
                        tf.math.subtract(
                            tf.math.subtract(self.phi_rec, self.phi_lo),
                            tf.math.multiply(self.C, self.dc),
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

        self.radical.assign(
            tf.math.subtract(
                tf.math.multiply(self.B, self.B),
                tf.math.multiply(
                    tf.math.multiply(tf.constant(3, dtype=tf.float64), self.A), self.C
                ),
            )
        )
        self.xmin.assign(
            tf.math.add(
                self.a_lo,
                tf.math.divide(
                    tf.math.add(tf.math.negative(self.B), tf.sqrt(self.radical)),
                    tf.math.multiply(tf.constant(3, dtype=tf.float64), self.A),
                ),
            )
        )

        self.a_j.assign(
            tf.where(tf.math.is_finite(self.xmin), self.xmin, self.none_variable)
        )
    
    

    def _quadmin(self):
        self.D.assign(self.phi_lo)
        self.C.assign(self.derphi_lo)
        self.db.assign(
            tf.math.subtract(
                self.a_hi,
                tf.math.multiply(self.a_lo, tf.constant(1.0, dtype=tf.float64)),
            )
        )
        self.B.assign(
            tf.where(
                tf.math.is_finite(self.db),
                tf.math.divide(
                    (
                        tf.math.subtract(
                            self.phi_hi,
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
                self.a_lo,
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
