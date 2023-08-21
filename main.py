import tensorflow as tf
import wandb
import argparse
from pathlib import Path
import pprint
import os
import dataclasses
from datetime import datetime
from timeit import default_timer as timer

# from dotenv import find_dotenv, load_dotenv

from src.configs.configs import DataConfig, OptimizerConfig, TrainConfig
import src.data.get_data as get_data
import src.models.model_archs as model_archs
from src.optimizer.get_optimizer import fetch_optimizer
from src.optimizer.cg_optimizer_eager import NonlinearCGEager
from src.optimizer.cg_optimizer_all_assign import NonlinearCG
from src.models.tracking import TrainTrack, CustomTqdmCallback, compute_full_loss
from src.configs import experiment_configs
from src.utils import setup
from src.models.new_resnet import resnet50


pp = pprint.PrettyPrinter(underscore_numbers=True).pprint
# load_dotenv(find_dotenv())


def main(
    run_eagerly: bool,
    data_config: DataConfig,
    optimizer_config: OptimizerConfig,
    train_config: TrainConfig,
):
    # set seet for replication
    #tf.debugging.set_log_device_placement(True)
    tf.random.set_seed(train_config.seed)
    
    pp(f'args profiler: ${args.profiler}')
    pp(f'args eagerly: ${args.run_eagerly}')
    
    # setup graph tracing
    if args.profiler:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        logdir = 'logs/func/%s' % stamp
        writer = tf.summary.create_file_writer(logdir)
        tf.summary.trace_on(graph=True, profiler=True)

    # Fetch all data, load to cache
    train_data, test_data = get_data.fetch_data(data_config)
    # train_data = train_data.cache()
    train_data = train_data.batch(
        batch_size=train_config.batch_size
        if train_config.batch_size
        else train_data.cardinality(),
        drop_remainder=True
    )

    if not args.profiler:
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
    if "MNIST" in data_config.name:
        model = model_archs.basic_cnn(data_config.num_classes)
    elif "CIFAR" in data_config.name:
        # model = resnet50()
        #model = model_archs.cifar_cnn(data_config.num_classes)
        # model = model_archs.resnet_18(data_config.num_classes)
        model = model_archs.resnet_18(data_config.num_classes)
    elif "IMAGENET" in data_config.name:
        model = model_archs.resnet_18(data_config.num_classes)
    else:
        pp(f"no specific model defined for the given dataset")

    model.build(input_shape=data_config.input_shape)
    model.summary()
    # Load chosen optimizer
    optimizer = fetch_optimizer(optimizer_config, model, train_config.loss_fn)
    
    if isinstance(optimizer, NonlinearCGEager):
        tf.config.run_functions_eagerly(args.run_eagerly)
        model.compile(
            loss=train_config.loss_fn,
            optimizer=optimizer,
            metrics=["accuracy"],
            run_eagerly=args.run_eagerly,
        )    
    
    if isinstance(optimizer, NonlinearCG):
        tf.config.run_functions_eagerly(args.run_eagerly)
        model.compile(
            loss=train_config.loss_fn,
            optimizer=optimizer,
            metrics=["accuracy"],
            run_eagerly=args.run_eagerly,
        )
    else:
        tf.config.run_functions_eagerly(args.run_eagerly)
        model.compile(
            loss=train_config.loss_fn,
            optimizer=optimizer,
            metrics=["accuracy"],
            run_eagerly=args.run_eagerly,
        )
        pp(optimizer)

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

    @tf.function
    def _train_step(x, y):
        with tf.GradientTape() as tape:
            preds = model(x)
            loss = train_config.loss_fn(y_true=y, y_pred=preds)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss
    
    @tf.function()
    def _nlcg_train_step(x, y):
        optimizer.apply_gradients(model.trainable_variables, x, y)
    
    
    if args.profiler:
        if isinstance(optimizer, NonlinearCGEager) or isinstance(optimizer, NonlinearCG):
            pp(f'Tracing NLCG')
            for epoch in range(train_config.max_epochs):
                for idx, (x, y) in enumerate(train_data):
                    _nlcg_train_step(x, y)
            with writer.as_default():
                tf.summary.trace_export(
                    name="nlcg_graph_trace",
                    step=3,
                    profiler_outdir=logdir)
        else:
            pp(f'Tracing RMSPROP/ADAM')
            for epoch in range(train_config.max_epochs):
                for idx, (x, y) in enumerate(train_data):
                    loss = _train_step(x, y)
            with writer.as_default():
                tf.summary.trace_export(
                    name="others_graph_trace",
                    step=3,
                    profiler_outdir=logdir)
    else:
        with CustomTqdmCallback(desc="Keras Optimizer", total=train_config.max_epochs) as t:
            for epoch in range(train_config.max_epochs):
                tracker.epoch += 1
                # Iterate through batches, calc gradients, update weights
                if isinstance(optimizer, NonlinearCGEager) or isinstance(optimizer, NonlinearCG):
                    epoch_loss = tf.keras.metrics.Mean()
                    for idx, (x, y) in enumerate(train_data):
                        # tracer start
                        # tf.profiler.experimental.start('logdir_path', options = options)
                        # optimizer step
                        _nlcg_train_step(x,y)
                        # tracer stop
                        # tf.profiler.experimental.stop()
                        tracker.nb_function_calls = optimizer.objective_tracker
                        tracker.nb_gradient_calls = optimizer.grad_tracker
                        loss = epoch_loss.update_state(
                            train_config.loss_fn(y_true=y, y_pred=model(x))
                        )
                        t.update_to(
                            loss=float(epoch_loss.result()),
                            steps=tracker.steps,
                            batch=idx + 1,
                        )
                else:
                    epoch_loss = tf.keras.metrics.Mean()
                    for idx, (x, y) in enumerate(train_data):
                        loss = _train_step(x, y)
                        epoch_loss.update_state(loss)
                        tracker.nb_function_calls += 1
                        tracker.nb_gradient_calls += 1
                        t.update_to(
                            loss=float(epoch_loss.result()),
                            steps=tracker.steps,
                            batch=idx + 1,
                        )

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
    """
    parser.add_argument(
        "--model", 
        required = True, 
        choices = list(m for m in experiment_configs.models.keys()), 
        help = "Model to train",
    )
    """

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
        "--gpu",
        required=False,
        choices=list(experiment_configs.gpus),
        help="Choose GPU to use e.g. ['0']",
        default=None,
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu
    
    tf.config.list_physical_devices('GPU')[0]
    
    setup.set_dtype(args.dtype)

    data_config = experiment_configs.data[args.data]
    data_config.path = Path(args.path)

    optimizer_config = experiment_configs.optimizers[args.optimizer]

    train_config = experiment_configs.train[data_config.task]
    train_config.batch_size = args.batch_size

    experiment_name = f"{data_config.name}-{optimizer_config.name}-{args.dtype}-eagerly-{args.run_eagerly}"

    pp(f"Experiment: {experiment_name}")
    pp(data_config)
    pp(optimizer_config)
    pp(train_config)

    if not args.use_wandb:
        os.environ["WANDB_MODE"] = "offline"

    wandb.init(
        project=os.getenv("WANDB_PROJECT", None),
        entity=os.getenv("WANDB_ENTITY", None),
        name=f"{experiment_name}",
        config={
            "model_name": f"CNN",
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
        optimizer_config=optimizer_config,
        train_config=train_config,
    )
