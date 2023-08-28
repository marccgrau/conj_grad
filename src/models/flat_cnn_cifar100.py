import tensorflow as tf


class FlatCNNCifar100(tf.keras.Model):
    def __init__(
        self,
        num_classes=100,
        num_base_filters=400,
        model_size="small",
        seed=42,
        **kwargs
    ):
        super(FlatCNNCifar100, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        self.scaling = 1.5 if model_size == "large" else 1
        self.conv1 = tf.keras.layers.Conv2D(
            filters=num_base_filters * self.scaling,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            kernel_initializer=self.initializer,
            bias_initializer=tf.zeros_initializer(),
        )
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(
            filters=(num_base_filters / 2) * self.scaling,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            kernel_initializer=self.initializer,
            bias_initializer=tf.zeros_initializer(),
        )
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = tf.keras.layers.Conv2D(
            filters=(num_base_filters / 4) * self.scaling,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
            kernel_initializer=self.initializer,
            bias_initializer=tf.zeros_initializer(),
        )
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(
            units=num_classes,
            activation="softmax",
            kernel_initializer=self.initializer,
            bias_initializer=tf.zeros_initializer(),
        )

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
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        output = self.fc1(x)

        return output
