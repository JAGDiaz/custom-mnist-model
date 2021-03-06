import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import matplotlib.pyplot as plt

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

    mnist = keras.datasets.mnist

    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    curr_dir = os.getcwd()
    mnist_weights_folder = os.path.join(curr_dir, "weights_folder")
    if not os.path.exists(mnist_weights_folder):
        os.makedirs(mnist_weights_folder)

    neurons_in_dense = [10*ii for ii in range(5, 30)]
    losses_n_accuracies = []
    for neuron_number in neurons_in_dense:
        particular_folder = os.path.join(mnist_weights_folder, f"num_neurons_{neuron_number:04d}")
        if not os.path.exists(particular_folder):
            os.makedirs(particular_folder)


        weight_callback = SaveWeightsAtEpoch(particular_folder)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(input_shape=(28, 28), name="Flattener"),
            tf.keras.layers.Dense(neuron_number, activation='relu', name="Dense"),
            tf.keras.layers.Dropout(0.2, name="Dropout"),
            tf.keras.layers.Dense(10, activation='softmax', name="Output")])

        model.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

        model.fit(x_train, y_train, epochs=5, callbacks=[weight_callback], shuffle=True, batch_size=500, verbose=0)
        result = model.evaluate(x_test, y_test, batch_size=500, verbose=0)
        losses_n_accuracies.append(result)

losses_n_accuracies = np.array(losses_n_accuracies)
fig, ax = plt.subplots(2,figsize=(10,10), sharex='col')

ax[0].plot(neurons_in_dense, losses_n_accuracies[...,0])
ax[0].grid(True, which='both')
ax[0].set_ylabel("Test Loss")
ax[0].set_yscale('log')


ax[1].plot(neurons_in_dense, 100*losses_n_accuracies[...,1])
ax[1].grid(True, which='both')
ax[1].set_ylabel("\% Accuracy")
ax[1].set_xlabel("Number of Neurons")

fig.tight_layout()
fig.savefig("loss_accuracy_by_neuron.png")
plt.close(fig)