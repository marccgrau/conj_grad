import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from src.models.model_archs import basic_cnn
from src.optimizer.cg_optimizer import NonlinearCG
import src.data.get_data as get_data
from src.configs import experiment_configs
from pathlib import Path
keras = tf.keras
K = keras.backend

if tf.config.list_physical_devices('GPU'):
  print("TensorFlow **IS** using the GPU")
else:
  print("TensorFlow **IS NOT** using the GPU")

data_config = experiment_configs.data["MNIST"]
data_config.path = Path('data')

train_data, test_data = get_data.fetch_data(data_config)
train_data = train_data.cache()
train_data = train_data.batch(
    batch_size=100
)
train_data = train_data.prefetch(tf.data.AUTOTUNE)

if test_data is not None:
    test_data = test_data.batch(
        batch_size=100
    )
    test_data = test_data.cache()
    test_data = test_data.prefetch(tf.data.AUTOTUNE)

model = basic_cnn(num_classes=10)
model.build(input_shape=(1, 28, 28, 1))

    
loss = keras.losses.CategoricalCrossentropy()

optimizer = NonlinearCG(model, loss)

model.compile(loss = loss, optimizer = optimizer, metrics = ['accuracy'])

model.fit(train_data, epochs=5)