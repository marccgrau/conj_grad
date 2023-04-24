import tensorflow as tf
import tensorflow.python.platform.build_info as build

print(tf.config.list_physical_devices('GPU'))
print(build.build_info)