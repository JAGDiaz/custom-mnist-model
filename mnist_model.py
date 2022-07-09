from gc import callbacks
import tensorflow as tf
from tensorflow import keras
import os

class SaveWeightsAtEpoch(keras.callbacks.Callback):
    """
    This callback (to be tested) is meant to save the weights of the 
    model into an HD5 file at the end of each training epoch. All my analysis
    for my thesis relies on saving the weights in this way, and the standard 
    checkpoint callback doesn't seem to support what I'm after (if it does, 
    it isn't obvious to me how you'd do that).
    """
    
    def __init__(self, save_folder):
        super(SaveWeightsAtEpoch, self).__init__()

        self.save_folder = save_folder

    def on_epoch_end(self, epoch, logs=None):
        file_to_save = os.path.join(self.save_folder,f"epoch_{epoch+1:05d}.h5")
        self.model.save_weights(file_to_save)



if __name__ == "__main__":

    curr_dir = os.getcwd()
    mnist_weights_folder = os.path.join(curr_dir, "weights_folder")

    if not os.path.exists(mnist_weights_folder):
        os.makedirs(mnist_weights_folder)

    weight_callback = SaveWeightsAtEpoch(mnist_weights_folder)

    mnist = keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=5, callbacks=[weight_callback], shuffle=True, batch_size=500)
    model.evaluate(x_test, y_test, verbose=2)


