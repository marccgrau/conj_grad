import tensorflow as tf


class FlatMLP(tf.keras.Model):
    def __init__(self, num_classes=10, num_units=128, **kwargs):
        super(FlatMLP, self).__init__(**kwargs)
        self.flatten = tf.keras.layers.Flatten()
        self.hidden = tf.keras.layers.Dense(units=num_units, activation="relu")
        self.output_layer = tf.keras.layers.Dense(
            units=num_classes, activation="softmax"
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
