import tensorflow as tf
from tensorflow import keras
import os

class SaveWeightsAtEpoch(keras.callbacks.Callback):
    """
    This callback (to be tested) is meant to save the weights of the 
    model into an HD5 file at the end of each training epoch. All my analysis
    for my thesis relies on saving the weights in this way, and the standard 
    checkpoint callback doesn't seem to support what I'm after (if it does, 
    it isn't obvious to me).
    """
    
    def __init__(self, save_folder):
        super(SaveWeightsAtEpoch, self).__init__()

        self.save_folder = save_folder

    def on_epoch_end(self, epoch, logs=None):
        file_to_save = os.path.join(self.save_folder,f"epoch_{epoch:05d}.h5")
        self.model.save_weights(file_to_save)
