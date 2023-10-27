import tensorflow as tf


class FlatMLP(tf.keras.Model):
    # Choose between small and large model dependent on experiment
    # Scaling is used to scale the number of filters in the model
    def __init__(
        self,
        num_classes=10,
        num_units_mlp=128,
        model_size="small",
        seed=42,
        **kwargs,
    ):
        super(FlatMLP, self).__init__(**kwargs)
        self.initializer = tf.keras.initializers.GlorotUniform(seed=seed)
        self.scaling = 2 if model_size == "large" else 1
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(
            units=num_units_mlp * self.scaling,
            activation="relu",
            kernel_initializer=self.initializer,
            bias_initializer=tf.zeros_initializer(),
        )
        self.output_layer = tf.keras.layers.Dense(
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

    def call(self, inputs):
        x = self.flatten(inputs)
        x = self.hidden(x)
        output = self.output_layer(x)
        return output
