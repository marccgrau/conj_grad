import tensorflow as tf
from src.models import model_archs
from src.optimizer.get_optimizer import fetch_optimizer
import src.data.get_data as get_data
from src.configs import experiment_configs
from pathlib import Path
from datetime import datetime
from src.utils import setup
import logging 

# Basic setup
setup.set_dtype("64")
if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

# Get configs
data_config = experiment_configs.data["CIFAR10"]
data_config.path = Path('/home/user/code/data/tf_data')
optimizer_config = experiment_configs.optimizers["NLCG"]
train_config = experiment_configs.train[data_config.task]

# Set seed
tf.random.set_seed(train_config.seed)

# Get data
train_data, test_data = get_data.fetch_data(data_config)
train_data = train_data.cache()
train_data = train_data.batch(
    batch_size=25000
)
train_data = train_data.prefetch(tf.data.AUTOTUNE)

if test_data is not None:
    test_data = test_data.batch(
        batch_size=1000
    )
    test_data = test_data.cache()
    test_data = test_data.prefetch(tf.data.AUTOTUNE)

# Load and build model
model = model_archs.cifar_cnn(data_config.num_classes)
model.build(input_shape=data_config.input_shape)
model.summary()

# Get optimizer
optimizer = fetch_optimizer(optimizer_config, model, train_config.loss_fn)

# Compile model with optimizer
tf.config.run_functions_eagerly(False)
model.compile(
    loss=train_config.loss_fn,
    optimizer=optimizer,
    metrics=["accuracy"],
    run_eagerly=False,
)

# logging
log_dir = "logs/func/" + datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_data, 
          epochs=4,
          validation_data=test_data,
          callbacks=[tensorboard_callback]
          )

print("Evaluate on test data")
results = model.evaluate(test_data, batch_size=1000)
print("test loss, test acc:", results)
