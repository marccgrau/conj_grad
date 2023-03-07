import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from keras.datasets import mnist
keras = tf.keras

keras.backend.set_floatx("float64")

@tf.function
def bin_crossentropy(y_true, y_pred):
    return tf.reduce_mean(
        keras.losses.binary_crossentropy(y_true=y_true, y_pred=y_pred, from_logits=True)
    )

class BasicCNN(tf.keras.Model):
    def __init__(self, num_classes, **kwargs):
        super(BasicCNN, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.conv3 = tf.keras.layers.Conv2D(
            filters=128,
            kernel_size=(3, 3),
            strides=1,
            padding="same",
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(units=256, activation="relu")
        self.fc2 = tf.keras.layers.Dense(units=num_classes, activation="softmax")
    
    def call(self, inputs, training=None, **kwargs):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        output = self.fc2(x)
        
        return output


def function_factory(model, loss, train_x, train_y):
    """A factory to create a function required by tfp.optimizer.lbfgs_minimize.
    Args:
        model [in]: an instance of `tf.keras.Model` or its subclasses.
        loss [in]: a function with signature loss_value = loss(pred_y, true_y).
        train_x [in]: the input part of training data.
        train_y [in]: the output part of training data.
    Returns:
        A function that has a signature of:
            loss_value, gradients = f(model_parameters).
    """
    
    # obtain the shapes of all trainable parameters in the model
    shapes = tf.shape_n(model.trainable_variables, out_type=tf.int32)
    n_tensors = len(shapes)

    # we'll use tf.dynamic_stitch and tf.dynamic_partition later, so we need to
    # prepare required information first
    count = 0
    idx = [] # stitch indices
    part = [] # partition indices

    for i, shape in enumerate(shapes):
        n = np.product(shape)
        idx.append(tf.reshape(tf.range(count, count+n, dtype=tf.int32), shape))
        part.extend([i]*n)
        count += n

    part = tf.constant(part)

    @tf.function
    def assign_new_model_parameters(params_1d):
        """A function updating the model's parameters with a 1D tf.Tensor.
        Args:
            params_1d [in]: a 1D tf.Tensor representing the model's trainable parameters.
        """

        params = tf.dynamic_partition(params_1d, part, n_tensors)
        for i, (shape, param) in enumerate(zip(shapes, params)):
            model.trainable_variables[i].assign(tf.reshape(param, shape))

    # now create a function that will be returned by this factory
    @tf.function
    def f(params_1d):
        """A function that can be used by tfp.optimizer.lbfgs_minimize.
        This function is created by function_factory.
        Args:
           params_1d [in]: a 1D tf.Tensor.
        Returns:
            A scalar loss and the gradients w.r.t. the `params_1d`.
        """

        # use GradientTape so that we can calculate the gradient of loss w.r.t. parameters
        with tf.GradientTape() as tape:
            # update the parameters in the model
            assign_new_model_parameters(params_1d)
            # calculate the loss
            loss_value = loss(model(train_x, training=True), train_y)

        # calculate gradients and convert to 1D tf.Tensor
        grads = tape.gradient(loss_value, model.trainable_variables)
        grads = tf.dynamic_stitch(idx, grads)
        avg_loss = tf.math.reduce_mean(loss_value)

        # print out iteration & loss
        f.iter.assign_add(1)
        tf.print("Iter:", f.iter, "loss:", avg_loss)

        # store loss value so we can retrieve later
        tf.py_function(f.history.append, inp=[avg_loss], Tout=[])

        return avg_loss, grads

    # store these information as members so we can use them outside the scope
    f.iter = tf.Variable(0)
    f.idx = idx
    f.part = part
    f.shapes = shapes
    f.assign_new_model_parameters = assign_new_model_parameters
    f.history = []

    return f

def _load_mnist():
    """
    Loads the MNIST dataset.
    @param data_config:
    @return: x_train, y_train, x_test, y_test
    """
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.reshape(x_train, (60000, 28, 28, 1)).astype("float64")
    x_test = np.reshape(x_test, (10000, 28, 28, 1)).astype("float64")
    x_train /= 255
    x_test /= 255
    y_train = keras.utils.to_categorical(y_train, 10).astype("float64")
    y_test = keras.utils.to_categorical(y_test, 10).astype("float64")
    assert x_train.shape == (60000, 28, 28, 1)
    assert x_test.shape == (10000, 28, 28, 1)
    assert y_train.shape == (60000, 10)
    assert y_test.shape == (10000, 10)
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    if all(v is not None for v in (x_test, y_test)):
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    else:
        ds_test = None
    #return ds_train, ds_test
    #x_train = tf.data.Dataset.from_tensor_slices(x_train)
    #y_train = tf.data.Dataset.from_tensor_slices(y_train)
    return x_train, y_train, x_test, y_test
    
if __name__ == "__main__":

    # use float64 by default
    tf.keras.backend.set_floatx("float64")
    
    # prepare training data
    x_1d = np.linspace(-1., 1., 11)
    x1, x2 = np.meshgrid(x_1d, x_1d)
    inps = np.stack((x1.flatten(), x2.flatten()), 1)
    outs = np.reshape(inps[:, 0]**2+inps[:, 1]**2, (x_1d.size**2, 1))

    x_train, y_train, x_test, y_test = _load_mnist()
    
    x_train = x_train[0:100]
    y_train = y_train[0:100]   
    x_test = x_test[0:100]
    y_test = y_test[0:100] 
    #x_train = x_train.batch(1)
    #y_train = y_train.batch(1)
    
    #x_train = tf.reshape(inps, [-1])
    #y_train = tf.reshape(outs, [-1])
    
    model = tf.keras.Sequential(
        [tf.keras.Input((28,28,1)),
         tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding="same", kernel_initializer='glorot_uniform',bias_initializer='zeros'),
         #tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding="same", kernel_initializer='glorot_uniform',bias_initializer='zeros'),
         #tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding="same", kernel_initializer='glorot_uniform',bias_initializer='zeros'),
         #tf.keras.layers.BatchNormalization(),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(units=256, activation="relu", kernel_initializer='glorot_uniform',bias_initializer='zeros'),
         tf.keras.layers.Dense(units=10, activation="softmax", kernel_initializer='glorot_uniform',bias_initializer='zeros')])

    # prepare prediction model, loss function, and the function passed to L-BFGS solver
    pred_model = tf.keras.Sequential(
        [tf.keras.Input(shape=[2,]),
         tf.keras.layers.Dense(64, "tanh"),
         tf.keras.layers.Dense(64, "tanh"),
         tf.keras.layers.Dense(1, None)])
    
    #loss_fun = tf.keras.losses.MeanSquaredError()
    loss = tf.keras.losses.categorical_crossentropy
    #func = function_factory(pred_model, loss_fun, inps, outs)
    func = function_factory(model, loss, x_train, y_train)
    
    # convert initial model parameters to a 1D tf.Tensor
    #init_params = tf.dynamic_stitch(func.idx, pred_model.trainable_variables)
    init_params = tf.dynamic_stitch(func.idx, model.trainable_variables)

    # train the model with L-BFGS solver
    results = tfp.optimizer.bfgs_minimize(
        value_and_gradients_function=func, initial_position=init_params, max_iterations=100
        )

    # after training, the final optimized parameters are still in results.position
    # so we have to manually put them back to the model
    func.assign_new_model_parameters(results.position)
    
    # do some prediction
    pred_outs = model.predict(x_test)
    
    correct_prediction = tf.equal(tf.argmax(pred_outs,1), tf.argmax(y_test,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(f'Accuracy: {accuracy}') 
    print(f'Example predictions: {pred_outs[0:9]}')
    print(f'Correct predictions: {correct_prediction}')
    
    
