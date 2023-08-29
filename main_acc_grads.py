import tensorflow as tf
import wandb
import argparse
from pathlib import Path
import pprint
import os
import dataclasses
import numpy as np
import warnings
from tensorflow.keras import backend as K

from src.configs.configs import DataConfig, OptimizerConfig, TrainConfig, ModelConfig
import src.data.get_data as get_data
from src.models.model_archs import get_model
from src.optimizer.get_optimizer import fetch_optimizer
from src.models.tracking import TrainTrack, CustomTqdmCallback, compute_full_loss
from src.configs import experiment_configs
from src.utils import setup
from src.utils.custom import as_Kfloat
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


pp = pprint.PrettyPrinter(underscore_numbers=True).pprint

amax = 1.0
c1 = 0.0001
c2 = 0.9
max_iters = 10
tol = 10e-7


def main(
    run_eagerly: bool,
    data_config: DataConfig,
    model_config: ModelConfig,
    train_config: TrainConfig,
):
    pp(f"args profiler: ${args.profiler}")
    pp(f"args eagerly: ${args.run_eagerly}")

    # Fetch all data, load to cache
    train_data, test_data = get_data.fetch_data(data_config)
    # train_data = train_data.cache()
    train_data = train_data.batch(
        batch_size=train_config.batch_size
        if train_config.batch_size
        else train_data.cardinality(),
        drop_remainder=True,
    )

    train_data = train_data.prefetch(tf.data.AUTOTUNE)

    if test_data is not None:
        test_data = test_data.batch(
            batch_size=train_config.batch_size
            if train_config.batch_size
            else test_data.cardinality()
        )
        test_data = test_data.cache()
        test_data = test_data.prefetch(tf.data.AUTOTUNE)

        # Load model architecture
    model = get_model(
        model_name=model_config.name,
        num_classes=data_config.num_classes,
        num_units_mlp=data_config.num_units_mlp,
        num_base_filters=data_config.num_base_filters,
        model_size=model_config.size,
        seed=train_config.seed,
    )

    model.build(input_shape=data_config.input_shape)
    model.summary()

    model.compile(
        loss=train_config.loss_fn,
        metrics=["accuracy"],
        run_eagerly=run_eagerly,
    )

    _weight_shapes = tf.shape_n(model.trainable_variables)
    _n_weights = len(_weight_shapes)
    _weight_indices = []
    _weight_partitions = []

    param_count = 0
    for i, shape in enumerate(_weight_shapes):
        n_params = tf.reduce_prod(shape)
        _weight_indices.append(
            tf.reshape(
                tf.range(param_count, param_count + n_params, dtype=tf.int32), shape
            )
        )
        _weight_partitions.extend(tf.ones(shape=(n_params,), dtype=tf.int32) * i)
        param_count += n_params

    loss_fn = train_config.loss_fn

    # Initiate trainings tracker
    tracker = TrainTrack()

    tracker.loss = compute_full_loss(
        model=model,
        loss_fn=train_config.loss_fn,
        data=train_data,
    )
    if test_data is not None:
        tracker.val_loss = compute_full_loss(
            model=model,
            loss_fn=train_config.loss_fn,
            data=test_data,
        )

    wandb.run.log(tracker.to_dict("log"))

    with CustomTqdmCallback(desc="Keras Optimizer", total=train_config.max_epochs) as t:
        for epoch in range(train_config.max_epochs):
            tracker.epoch += 1
            epoch_loss = tf.keras.metrics.Mean()
            model = update_step(
                model,
                loss_fn,
                train_data,
                _weight_shapes,
                _weight_partitions,
                _n_weights,
                _weight_indices,
            )
            total_loss = 0
            for idx, (x, y) in enumerate(train_data):
                batch_loss = loss_fn(y, model(x, training=False))
                total_loss += batch_loss
            epoch_loss.update_state(total_loss)

            # Compute metrics for all defined metrics
            tracker.loss = epoch_loss.result()
            for metric_fn in train_config.metrics:
                metric_name = f"train_{metric_fn.__name__.split('.')[-1]}"
                metric = compute_full_loss(
                    model=model,
                    loss_fn=metric_fn,
                    data=train_data,
                )
                tracker.metrics[metric_name] = metric

            # Log best epoch
            if tf.less(tracker.loss, tracker.best_loss):
                tracker.best_loss = tracker.loss
                tracker.best_epoch = tracker.epoch
                wandb.run.summary.update(tracker.to_dict("summary"))
                t.update_to(best_loss=float(tracker.best_loss))

            # Compute metrics for test data
            if test_data is not None:
                tracker.val_loss = compute_full_loss(
                    model=model,
                    loss_fn=train_config.loss_fn,
                    data=test_data,
                )
                if tf.less(tracker.val_loss, tracker.best_val_loss):
                    tracker.best_val_loss = tracker.val_loss
                    tracker.best_val_epoch = tracker.epoch
                    wandb.run.summary.update(tracker.to_dict("summary"))
                for metric_fn in train_config.metrics:
                    metric_name = f"test_{metric_fn.__name__.split('.')[-1]}"
                    metric = compute_full_loss(
                        model=model,
                        loss_fn=metric_fn,
                        data=test_data,
                    )
                    tracker.metrics[metric_name] = metric
                t.update_to(val_loss=tracker.val_loss, **tracker.metrics)
                wandb.run.summary.update(tracker.to_dict("summary"))

            # Log tracker
            wandb.run.log(tracker.to_dict("log"))
            t.update_to(tracker.epoch)

            # Stop training if max calls reached
            if tracker.steps >= train_config.max_calls:
                break


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--path", required=True, type=str, help="Path to data directory"
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=list(m for m in experiment_configs.models.keys()),
        help="Model to train",
    )

    parser.add_argument(
        "--model_size",
        required=True,
        choices=["small", "large"],
        help="Size of the model (small or large)",
    )

    parser.add_argument(
        "--data",
        required=True,
        choices=list(c for c in experiment_configs.data.keys()),
        help="Dataset to use",
    )

    parser.add_argument(
        "--dtype",
        required=True,
        choices=list(experiment_configs.dtypes),
        help="Data precision to use",
    )

    parser.add_argument(
        "--optimizer",
        required=True,
        choices=list(o for o in experiment_configs.optimizers.keys()),
        help="Optimizer to use",
    )

    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--run_eagerly",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--profiler",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        required=False,
        help="Optional batch size if not on full dataset",
        default=None,
    )

    parser.add_argument(
        "--max_epochs",
        type=int,
        required=False,
        help="Optional number of max epochs",
        default=None,
    )

    parser.add_argument(
        "--gpu",
        required=False,
        choices=list(experiment_configs.gpus),
        help="Choose GPU to use e.g. ['0']",
        default=None,
    )

    return parser.parse_args()


def _update_model_parameters(
    model, new_params, _weight_shapes, _weight_partitions, _n_weights
):
    """
    Assign new set of weights to model
    Parameters
    ----------
    new_params: tf.Tensor
        New model weights
    Returns
    -------

    """
    params = tf.dynamic_partition(
        data=new_params, partitions=_weight_partitions, num_partitions=_n_weights
    )
    for i, (param, shape) in enumerate(zip(params, _weight_shapes)):
        param = tf.reshape(param, shape)
        param = tf.cast(param, dtype=K.floatx())
        model.trainable_variables[i].assign(param)


def _set_trial_model_params(
    model,
    initial_weights,
    alpha,
    search_direction,
    _weight_shapes,
    _weight_partitions,
    _n_weights,
    _weight_indices,
):
    """
    Assign new set of weights to model
    Parameters
    ----------
    new_params: tf.Tensor
        New model weights
    Returns
    -------

    """
    model_params = tf.dynamic_stitch(indices=_weight_indices, data=initial_weights)
    new_params = model_params + alpha * search_direction
    params = tf.dynamic_partition(
        data=new_params, partitions=_weight_partitions, num_partitions=_n_weights
    )
    for i, (param, shape) in enumerate(zip(params, _weight_shapes)):
        param = tf.reshape(param, shape)
        param = tf.cast(param, dtype=K.floatx())
        model.trainable_variables[i].assign(param)

    return model


def _objective_call(model, loss_fn, train_data):
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
    epoch_loss = 0
    for idx, (x, y) in enumerate(train_data):
        batch_loss = loss_fn(y, model(x, training=False))
        epoch_loss += batch_loss
    return epoch_loss


def _gradient_call(model, loss_fn, train_data, _weight_indices):
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
    total_gradients = [
        tf.Variable(tf.zeros_like(w), trainable=False)
        for w in model.trainable_variables
    ]
    num_batches = 0
    for idx, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            batch_loss = loss_fn(y, y_pred)
        grads = tape.gradient(batch_loss, model.trainable_variables)
        for i, (grad, total_gradient) in enumerate(zip(grads, total_gradients)):
            total_gradient.assign_add(grad)
        num_batches += 1
    average_gradient = [
        grad / tf.cast(num_batches, dtype=tf.float64) for grad in total_gradients
    ]
    return _from_matrices_to_vector(average_gradient, _weight_indices)


def _obj_func_and_grad_call(model, loss_fn, train_data, _weight_indices):
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
    epoch_loss = 0
    total_gradients = [
        tf.Variable(tf.zeros_like(w), trainable=False) for w in model.trainable_weights
    ]
    num_batches = 0
    for idx, (x, y) in enumerate(train_data):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            batch_loss = loss_fn(y, y_pred)
        grads = tape.gradient(batch_loss, model.trainable_variables)
        epoch_loss += batch_loss
        for i, (grad, total_gradient) in enumerate(zip(grads, total_gradients)):
            total_gradient.assign_add(grad)
        num_batches += 1
    average_gradient = [
        grad / tf.cast(num_batches, dtype=tf.float64) for grad in total_gradients
    ]
    return epoch_loss, _from_matrices_to_vector(average_gradient, _weight_indices)


def update_step(
    model,
    loss_fn,
    train_data,
    _weight_shapes,
    _weight_partitions,
    _n_weights,
    _weight_indices,
):
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
    iters = 0
    obj_val, grad = _obj_func_and_grad_call(model, loss_fn, train_data, _weight_indices)
    r = -grad
    d = r
    while iters < max_iters:
        initial_weights = model.trainable_variables
        # Perform line search to determine alpha_star
        alpha = wolfe_line_search(
            model=model,
            loss_fn=loss_fn,
            train_data=train_data,
            _weight_shapes=_weight_shapes,
            _weight_partitions=_weight_partitions,
            _n_weights=_n_weights,
            _weight_indices=_weight_indices,
            maxiter=50,
            search_direction=d,
        )
        logger.info(f"alpha after line search: {alpha}")
        # update weights along search directions
        if alpha is None:
            logger.info("Alpha is None. Making no step.")
            # w_new = weights + 10e-1 * r
            break
        else:
            model = _set_trial_model_params(
                model,
                initial_weights,
                alpha,
                d,
                _weight_shapes,
                _weight_partitions,
                _n_weights,
                _weight_indices,
            )
        # get new objective value and gradient
        obj_val_new, grad_new = _obj_func_and_grad_call(
            model, loss_fn, train_data, _weight_indices
        )
        # set r_{k+1}
        r_new = -grad_new
        # Calculate Polak-Ribiére beta
        beta = tf.reduce_sum(tf.multiply(r_new, r_new - r)) / tf.reduce_sum(
            tf.multiply(r, r)
        )
        # PRP+ with max{beta{PR}, 0}
        beta = np.maximum(beta, 0)
        logger.info(f"beta: {beta}")
        # Determine new search direction for next iteration step
        d_new = r_new + beta * d
        # Check for convergence
        if tf.reduce_sum(tf.abs(obj_val_new - obj_val)) < tol:
            logger.info("Stop NLCG, no sufficient decrease in value function")
            break
        # check if vector norm is smaller than the tolerance
        if tf.norm(r_new) <= tol:
            logger.info("Stop NLCG, gradient norm smaller than tolerance")
            break
        # Set new values as old values for next iteration step
        grad = grad_new
        r = r_new
        d = d_new
        obj_val = obj_val_new
        iters += 1
    return model


def _from_matrices_to_vector(matrices, _weight_indices):
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
    return tf.dynamic_stitch(indices=_weight_indices, data=matrices)


def wolfe_line_search(
    model,
    loss_fn,
    train_data,
    _weight_shapes,
    _weight_partitions,
    _n_weights,
    _weight_indices,
    maxiter=10,
    search_direction=None,
):
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
    initial_weights = model.trainable_variables
    trial_model = tf.keras.models.clone_model(model)
    trial_model.build(input_shape=data_config.input_shape)
    trial_model.compile(
        loss=train_config.loss_fn,
        metrics=["accuracy"],
        run_eagerly=True,
    )
    # copy the whole model
    trial_model.set_weights(model.get_weights())

    # Leaving the weights as is, is the equivalent of setting alpha to 0
    # Thus, we get objective value at 0 and the gradient at 0
    phi0 = _objective_call(trial_model, loss_fn, train_data)
    # We need the directional derivative at 0 following the Wolfe Conditions
    # Thus, we get the gradient at 0 and multiply it with the search direction
    derphi0 = tf.tensordot(
        _gradient_call(trial_model, loss_fn, train_data, _weight_indices),
        search_direction,
        1,
    )

    # Set alpha bounds
    alpha0 = 0.0
    alpha1 = 1.0
    # Optional setting of an alpha max, if defined
    if amax is not None:
        alpha1 = np.minimum(alpha1, amax)

    # get objective value at a new possible position, i.e. w_k + alpha1 * d_k
    trial_model = _set_trial_model_params(
        trial_model,
        initial_weights,
        alpha1,
        search_direction,
        _weight_shapes,
        _weight_partitions,
        _n_weights,
        _weight_indices,
    )
    phi_a1 = _objective_call(trial_model, loss_fn, train_data)

    # Initial alpha_lo equivalent to alpha = 0
    phi_a0 = phi0
    derphi_a0 = derphi0
    i = 0
    for i in range(maxiter):
        # tf.cond(initial_cond(), lambda: break_ = True, lambda: None)
        if alpha1 == 0.0 or alpha0 == amax:
            # if tf.math.logical_or(tf.math.equal(alpha1, tf.Variable(0.)), tf.math.equal(alpha0, amax)):
            alpha_star = None
            # phi_star = phi0
            # derphi_star = None

            if alpha1 == 0:
                # if tf.math.equal(alpha1, 0):
                warnings.warn("Rounding errors preventing line search from converging")
            else:
                warnings.warn(
                    f"Line search could not find solution less than or equal to {amax}"
                )
            break

        if phi_a1 > phi0 + c1 * alpha1 * derphi0 or (phi_a1 >= phi_a0 and i > 1):
            alpha_star = _zoom(
                trial_model,
                loss_fn,
                train_data,
                _weight_shapes,
                _weight_partitions,
                _n_weights,
                _weight_indices,
                alpha0,
                alpha1,
                phi_a0,
                phi_a1,
                derphi_a0,
                phi0,
                derphi0,
                search_direction,
            )
            break

        # Second condition: |phi'(alpha_i)| <= -c2 * phi'(0)
        trial_model = _set_trial_model_params(
            trial_model,
            initial_weights,
            alpha1,
            search_direction,
            _weight_shapes,
            _weight_partitions,
            _n_weights,
            _weight_indices,
        )
        derphi_a1 = tf.tensordot(
            _gradient_call(trial_model, loss_fn, train_data, _weight_indices),
            search_direction,
            1,
        )
        if np.absolute(derphi_a1) <= -c2 * derphi0:
            alpha_star = alpha1  # suitable alpha found set to star and return
            trial_model = _set_trial_model_params(
                trial_model,
                initial_weights,
                alpha_star,
                search_direction,
                _weight_shapes,
                _weight_partitions,
                _n_weights,
                _weight_indices,
            )
            phi_star = _objective_call(trial_model, loss_fn, train_data)
            derphi_star = derphi_a1
            break

        # Third condition: phi'(alpha_i) >= 0
        if derphi_a1 >= 0.0:
            # if (tf.math.greater_equal(derphi_a1, 0)):
            alpha_star = _zoom(
                trial_model,
                loss_fn,
                train_data,
                _weight_shapes,
                _weight_partitions,
                _n_weights,
                _weight_indices,
                alpha1,
                alpha0,
                phi_a1,
                phi_a0,
                derphi_a1,
                phi0,
                derphi0,
                search_direction,
            )
            break

        # extrapolation step of alpha_i as no conditions are met
        alpha2 = 2 * alpha1
        # check if we don't overshoot amax
        if amax is not None:
            alpha2 = np.minimum(alpha2, amax)

        # update values for next iteration to check conditions
        alpha0 = alpha1
        alpha1 = alpha2
        phi_a0 = phi_a1
        trial_model = _set_trial_model_params(
            trial_model,
            initial_weights,
            alpha1,
            search_direction,
            _weight_shapes,
            _weight_partitions,
            _n_weights,
            _weight_indices,
        )
        phi_a1 = _objective_call(trial_model, loss_fn, train_data)
        derphi_a0 = derphi_a1

    # if no break occurs, then we have not found a suitable alpha after maxiter
    else:
        alpha_star = alpha1
        warnings.warn("Line search did not converge")

    return alpha_star


def _zoom(
    model,
    loss_fn,
    train_data,
    _weight_shapes,
    _weight_partitions,
    _n_weights,
    _weight_indices,
    a_lo,
    a_hi,
    phi_lo,
    phi_hi,
    derphi_lo,
    phi0,
    derphi0,
    search_direction,
):
    """
    Zoom stage of approximate line search satisfying strong Wolfe conditions.
    """
    initial_weights = model.trainable_variables
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

        if i > 0:
            cchk = delta1 * dalpha
            a_j = _cubicmin(a_lo, phi_lo, derphi_lo, a_hi, phi_hi, a_rec, phi_rec)
        if (i == 0) or (a_j is None) or (a_j > b - cchk) or (a_j < a + cchk):
            qchk = delta2 * dalpha
            a_j = _quadmin(
                point_1=a_lo,
                obj_1=phi_lo,
                grad_1=derphi_lo,
                point_2=a_hi,
                obj_2=phi_hi,
            )
            if (a_j is None) or (a_j > b - qchk) or (a_j < a + qchk):
                a_j = a_lo + 0.5 * dalpha

        # Check new value of a_j
        model = _set_trial_model_params(
            model,
            initial_weights,
            a_j,
            search_direction,
            _weight_shapes,
            _weight_partitions,
            _n_weights,
            _weight_indices,
        )
        phi_aj = _objective_call(model, loss_fn, train_data)
        if (phi_aj > phi0 + c1 * a_j * derphi0) or (phi_aj >= phi_lo):
            phi_rec = phi_hi
            a_rec = a_hi
            a_hi = a_j
            phi_hi = phi_aj
        else:
            model = _set_trial_model_params(
                model,
                initial_weights,
                a_j,
                search_direction,
                _weight_shapes,
                _weight_partitions,
                _n_weights,
                _weight_indices,
            )
            derphi_aj = tf.tensordot(
                _gradient_call(model, loss_fn, train_data, _weight_indices),
                search_direction,
                1,
            )
            if tf.math.abs(derphi_aj) <= -c2 * derphi0:
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
            a_star = None
            break
    return a_star


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
        d1 = np.empty((2, 2))
        d1[0, 0] = dc**2
        d1[0, 1] = -(db**2)
        d1[1, 0] = -(dc**3)
        d1[1, 1] = db**3
        # TODO: Hardcoded dtype
        [A, B] = np.dot(
            d1,
            np.asarray([fb - fa - C * db, fc - fa - C * dc], dtype="float64").flatten(),
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


def _quadmin(point_1, obj_1, grad_1, point_2, obj_2):
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


if __name__ == "__main__":
    args = parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        # Restrict TensorFlow to only use the first GPU
        try:
            tf.config.set_visible_devices(gpus[0], "GPU")
            logical_gpus = tf.config.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        except RuntimeError as e:
            # Visible devices must be set before GPUs have been initialized
            print(e)

    print(gpus)

    setup.set_dtype(args.dtype)

    data_config = experiment_configs.data[args.data]
    data_config.path = Path(args.path)

    model_config = experiment_configs.models[args.model]
    model_config.size = args.model_size.lower()

    optimizer_config = experiment_configs.optimizers[args.optimizer]

    train_config = experiment_configs.train[data_config.task]
    train_config.batch_size = args.batch_size
    if args.max_epochs:
        train_config.max_epochs = args.max_epochs

    tf.random.set_seed(train_config.seed)

    experiment_name = f"TESTRUN-{data_config.name}-{model_config.name}-{model_config.size}\
        -{optimizer_config.name}-{args.dtype}-eagerly-{args.run_eagerly}"

    if "NLCG" in optimizer_config.name:
        experiment_name = f"TESTRUN-{data_config.name}-{model_config.name}-{model_config.size}\
                            -{optimizer_config.name}-{args.dtype}-eagerly-{args.run_eagerly}-\
                            MAXITERS-{optimizer_config.max_iters}"
    else:
        experiment_name = f"TESTRUN-{data_config.name}-{model_config.name}-{model_config.size}\
                            -{optimizer_config.name}-{args.dtype}-eagerly-{args.run_eagerly}"

    pp(f"Experiment: {experiment_name}")
    pp(data_config)
    pp(model_config)
    pp(optimizer_config)
    pp(train_config)

    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project=os.getenv("WANDB_PROJECT", None),
        entity=os.getenv("WANDB_ENTITY", None),
        name=f"{experiment_name}",
        config={
            "model_name": f"{model_config.name}-{model_config.size}",
            "dtype": f"{setup.DTYPE}",
            "data": dataclasses.asdict(data_config),
            "training": dataclasses.asdict(train_config),
            "optimizer": dataclasses.asdict(optimizer_config),
        },
    )

    pp(dict(wandb.config))

    main(
        run_eagerly=args.run_eagerly,
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
    )
