import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

def walk_h5(file_handle, space="", num_spaces=2):
    for key in file_handle.keys():
        print(space + key)
        try:
            walk_h5(file_handle[key], space=space + " "*num_spaces, num_spaces=num_spaces)
        except:
            pass

weights_folder = os.path.join(os.getcwd(), "weights_folder")

h5_files = [os.path.join(weights_folder, file) for file in os.listdir(weights_folder) if file.endswith('.h5')]

for file in h5_files:
    h5_file = h5py.File(file, 'r')

    print(file)
    walk_h5(h5_file)
    print()

    h5_file.close()