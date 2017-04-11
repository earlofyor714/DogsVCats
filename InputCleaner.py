import numpy as np

class InputCleaner:
    def image_to_array(self, image):
        return np.array(image)

    def scale_image(self, image, scale):
        return image/scale

    def condense(self, image):
        return image.reshape((1, image.size))

    def add_to_array(self, array, image):
        return np.append(array, image, axis=0)
