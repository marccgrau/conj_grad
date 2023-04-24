import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from src.models.model_archs import basic_cnn
from src.optimizer.cg_optimizer import NonlinearCG
from src.optimizer.cg_optimizer_eager import NonlinearCGEager
import src.data.get_data as get_data
from src.configs import experiment_configs
from pathlib import Path
import datetime
from src.utils import setup
import logging 



setup.set_dtype("64")
tf.config.run_functions_eagerly(True)

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

data_config = experiment_configs.data["MNIST"]
data_config.path = Path('data')

train_data, test_data = get_data.fetch_data(data_config)
train_data.take(10000)
train_data = train_data.cache()
train_data = train_data.batch(
    batch_size=1000
)
train_data = train_data.prefetch(tf.data.AUTOTUNE)

if test_data is not None:
    test_data = test_data.batch(
        batch_size=1000
    )
    test_data = test_data.cache()
    test_data = test_data.prefetch(tf.data.AUTOTUNE)


model = basic_cnn(num_classes=10)
model.build(input_shape=(1, 28, 28, 1))

model.summary()
    
loss = tf.keras.losses.CategoricalCrossentropy()

execution_mode = 'eager'
if execution_mode == 'eager':
    optimizer = NonlinearCGEager(model, loss)
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'], run_eagerly=True)
else:
    optimizer = NonlinearCG(model, loss)
    model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

# logging
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(train_data, 
          epochs=2,
          validation_data=test_data,
          callbacks=[tensorboard_callback]
          )

print("Evaluate on test data")
results = model.evaluate(test_data, batch_size=128)
print("test loss, test acc:", results)
