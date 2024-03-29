import tensorflow as tf

from src.models.res_blocks import make_basic_block_layer, make_bottleneck_layer


class ResNetTypeI(tf.keras.Model):
    def __init__(self, layer_params, num_classes):
        super(ResNetTypeI, self).__init__()

        self.conv1 = tf.keras.layers.Conv2D(
            filters=16, kernel_size=(7, 7), strides=2, padding="same"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=2, padding="same"
        )

        self.layer1 = make_basic_block_layer(num_filters=16, blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(
            num_filters=32, blocks=layer_params[1], stride=2
        )
        self.layer3 = make_basic_block_layer(
            num_filters=64, blocks=layer_params[2], stride=2
        )
        self.layer4 = make_basic_block_layer(
            num_filters=128, blocks=layer_params[3], stride=2
        )

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(
            units=num_classes, activation=tf.keras.activations.softmax
        )

    def train_step(self, data):
        # fetch data
        x, y = data
        # apply updates from optimizer
        # new_weights = self.optimizer.apply_gradients(self.trainable_weights, x, y)
        self.optimizer.apply_gradients(self.trainable_weights, x, y)
        # for var, new_value in zip(self.trainable_weights, new_weights):
        #    var.assign(new_value)
        # self.compiled_metrics.update_state(y, self(x, training=True))
        y_pred = self(x, training=True)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


class ResNetTypeII(tf.keras.Model):
    def __init__(self, layer_params, num_classes):
        super(ResNetTypeII, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=64, kernel_size=(7, 7), strides=2, padding="same"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(
            pool_size=(3, 3), strides=2, padding="same"
        )

        self.layer1 = make_bottleneck_layer(num_filters=64, blocks=layer_params[0])
        self.layer2 = make_bottleneck_layer(
            num_filters=128, blocks=layer_params[1], stride=2
        )
        self.layer3 = make_bottleneck_layer(
            num_filters=256, blocks=layer_params[2], stride=2
        )
        self.layer4 = make_bottleneck_layer(
            num_filters=512, blocks=layer_params[3], stride=2
        )

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(
            units=num_classes, activation=tf.keras.activations.softmax
        )

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output


class ResNetCIFAR(tf.keras.Model):
    def __init__(self, layer_params, num_classes):
        super(ResNetCIFAR, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=8, kernel_size=(3, 3), strides=1, padding="same"
        )

        self.layer1 = make_basic_block_layer(num_filters=8, blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(
            num_filters=16, blocks=layer_params[1], stride=2
        )
        self.layer3 = make_basic_block_layer(
            num_filters=32, blocks=layer_params[2], stride=2
        )

        self.avgpool = tf.keras.layers.GlobalAveragePooling2D()
        self.fc = tf.keras.layers.Dense(
            units=num_classes, activation=tf.keras.activations.softmax
        )
    
    def train_step(self, data):
        # fetch data
        x, y = data
        # apply updates from optimizer
        # new_weights = self.optimizer.apply_gradients(self.trainable_weights, x, y)
        self.optimizer.apply_gradients(self.trainable_weights, x, y)
        # for var, new_value in zip(self.trainable_weights, new_weights):
        #    var.assign(new_value)
        # self.compiled_metrics.update_state(y, self(x, training=True))
        y_pred = self(x, training=True)
        loss = self.compute_loss(y=y, y_pred=y_pred)
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
    
    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = tf.nn.relu(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.avgpool(x)
        x = tf.keras.layers.Flatten()(x)
        output = self.fc(x)

        return output
