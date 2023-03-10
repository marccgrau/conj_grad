import collections
import tensorflow as tf
from tensorflow_probability.python.internal import distribution_util
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.internal import dtype_util

from src.optimizer import nlcg_utils

NlcgOptimizerResults = collections.namedtuple(
    'NlcgOptimizerResults', [
        'converged',
        'failed',
        'num_iter',
        'num_objective_evaluations',
        'position',
        'objective_value',
        'objective_gradient',
        'search_direction',
    ])

def minimize(value_and_gradients_function,
             initial_position,
             tolerance=1e-14,
             f_relative_tolerance=1e-14,
             f_absolute_tolerance=1e-14,
             x_tolerance=1e-14,
             max_iter=100,
             max_line_search_iterations=10,
             parallel_iterations=1,
             stopping_condition=None,
             name=None,
             ):
    with tf.name_scope(name or 'minimize'):
        initial_position = tf.convert_to_tensor(initial_position, name='initial_position')
        dtype = dtype_util.base_dtype(initial_position.dtype)
        tolerance = tf.convert_to_tensor(tolerance, dtype=dtype, name='grad_tolerance')
        max_iter = tf.convert_to_tensor(max_iter, name='max_iter')
        
        input_shape = ps.shape(initial_position)
        batch_shape, domain_size = input_shape[:-1], input_shape[-1]
    
    if stopping_condition is None:
      stopping_condition = nlcg_utils.converged_all
    
    # Control inputs are an optional list of tensors to evaluate before
    # the start of the search procedure. These can be used to assert the
    # validity of inputs to the search procedure.
    control_inputs = None
        
    def _cond(state):
        # Condition for while loop to stop
        return (state.num_iter < max_iter)
    
    def _body(state):
        """Main optimization loop."""
        # check for termination
        state = nlcg_utils.terminate_if_not_finite(state)
        
        # set current state
        current_state = state
                
        # x_0 = initial_position, d_0 = r_0 = -grad(f(x_0))
        # otherwise set to search direction determined in previous step after line search
        # TODO: check for validity of first iteration
        old_r = -current_state.objective_gradient
        search_direction = current_state.search_direction
        
        # determine next state 
        # perform HZ line search returning x_{i+1} = x_i + alpha_i * d_i
        # further returned next objective value and objective gradient
        next_state = nlcg_utils.line_search_step(
            current_state, value_and_gradients_function, search_direction,
            tolerance, f_relative_tolerance, x_tolerance, stopping_condition,
            max_line_search_iterations, f_absolute_tolerance)
        
        # derive new search direction from next state 
        # r_{i+1} = -grad(f(x_{i+1}))
        new_r = -next_state.objective_gradient
        
        # derive beta_{i+1} from gradient of next state (depending on method not only gradient)
        # new_gradient = r_{i+1}, search_direction = r_0 for first iteration, old_gradient = r_{i} for i > 0
        # TODO: adjust for different input arguments depending on method
        beta = _get_beta('PR', new_r, old_r)
        
        # determine new search direction d_{i+1} = r_{i+1} + beta_{i+1} * d_{i}
        new_search_direction = _get_conjugate_search_direction(new_r, beta, search_direction)

        # TODO: update state variables such that new situation is reflected for next iteration
        # and make sure that it actually returns the new state in correct format
        new_state = nlcg_utils.update_fields(state,
                                 converged=next_state.converged,
                                 position = next_state.position,
                                 objective_value=next_state.objective_value,
                                 objective_gradient=next_state.objective_gradient,
                                 search_direction=new_search_direction)
        return [new_state]
                


    kwargs = nlcg_utils.get_initial_state_args(
        value_and_gradients_function,
        initial_position,
        tolerance,
        control_inputs)
    
    initial_state = NlcgOptimizerResults(**kwargs)
    # repeat body while cond is true
    return tf.while_loop(
        cond=_cond,
        body=_body,
        loop_vars=[initial_state],
        parallel_iterations=parallel_iterations)[0]


def _get_beta(method, grad_new, grad_old, old_search_direction = None):
    # grad_new = r_{i+1}
    # grad_old = r_{i}
    # TODO: consider to set beta = max(0, beta^{PR}) equivalent to restarting CG if beta < 0
    if method == 'PR':
        y_hat = grad_new - grad_old
        beta = tf.reduce_sum(tf.multiply(grad_new, y_hat)) / tf.reduce_sum(tf.multiply(grad_old, grad_old))
    elif method == 'HZ' and old_search_direction is not None:
            y_hat = grad_new - grad_old
            beta = tf.linalg.matmul(y_hat, grad_new) / tf.linalg.matmul(y_hat, old_search_direction)
            beta = beta - 2 * tf.linalg.matmul(y_hat, y_hat) * tf.linalg.matmul(old_search_direction, grad_new) / (tf.linalg.matmul(y_hat, old_search_direction) ** 2)
    else:
        raise ValueError('Invalid method! Try one of the followings: PR, HZ.')
    return beta

def _get_conjugate_search_direction(grad, beta, old_search_direction):
    # d_{i+1} = r_{i+1} + beta_{i+1} * d_{i}
    new_search_direction = grad + beta * old_search_direction 
    return new_search_direction




