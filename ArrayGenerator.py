import os
import numpy as np

import FileManager
import InputCleaner


class ArrayGenerator:
    def __init__(self):
        self.InputCleaner = InputCleaner.InputCleaner()

    def generate_inputs_labels(self, path, image_size):
        X = np.array([]).reshape(0, image_size*image_size*3) # should we pass in total size or calculate by hand?
        y = np.array([], dtype=int).reshape(0, 1)

        for filename in os.listdir(path):
            if '.jpg' in filename:
                print("{}".format(filename))
                file = path + "/" + filename
                image = FileManager.read(file)
                array = self.InputCleaner.image_to_array(image)
                FileManager.close(image)

                condensed_array = self.InputCleaner.condense(array)
                scaled_image = self.InputCleaner.scale_image(condensed_array, 255)
                X = self.InputCleaner.add_to_array(X, scaled_image)

                if "cat" in filename:
                    value = np.asarray([1]).reshape((1, 1))
                    y = self.InputCleaner.add_to_array(y, value)
                else:
                    value = np.asarray([0]).reshape((1, 1))
                    y = self.InputCleaner.add_to_array(y, value)

                # to do: create label array

        return X, y
