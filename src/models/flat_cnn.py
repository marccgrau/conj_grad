import tensorflow as tf


class FlatCNN(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(FlatCNN, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=num_classes, activation="softmax")

    def train_step(self, data):
        # fetch data
        x, y = data
        # apply updates from optimizer
        tempmodel = self.optimizer.apply_gradients(self.trainable_variables, x, y)
        self.set_weights(tempmodel.get_weights())
        self.compiled_metrics.update_state(y, self(x))
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        output = self.fc1(x)

        return output
