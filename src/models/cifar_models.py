import tensorflow as tf


class CIFARCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CIFARCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(1, kernel_size=(3, 3), activation="relu")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(2, kernel_size=(3, 3), activation="relu")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(32, activation="relu")
        self.fc2 = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        output = self.fc2(x)
        return output