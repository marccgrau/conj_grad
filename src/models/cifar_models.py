import tensorflow as tf


class CIFARCNN(tf.keras.Model):
    def __init__(self, num_classes):
        super(CIFARCNN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu")
        self.pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv2 = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu")
        self.pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.conv3 = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu")
        self.pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(256, activation="relu")
        self.fc2 = tf.keras.layers.Dense(128, activation="relu")
        self.fc3 = tf.keras.layers.Dense(num_classes, activation="softmax")

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        output = self.fc3(x)
        return output
