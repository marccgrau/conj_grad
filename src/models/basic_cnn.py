import tensorflow as tf

#TODO: Netz verkleinern
class BasicCNN(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(BasicCNN, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=3,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )
        #self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=6,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )
        #self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(
            filters=9,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )
        #self.bn3 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=8, activation="relu")
        self.fc2 = tf.keras.layers.Dense(units=num_classes, activation="softmax")
    
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
        #x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        #x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        #x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        output = self.fc2(x)
        
        return output