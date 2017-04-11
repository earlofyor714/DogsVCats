import FileManager
import numpy as np
from PIL import Image as Image


class ImageSquarer:

    def resizeImage(self, file, size_goal):
        image = FileManager.read(file)
        image_shape = np.array(image).shape

        if image_shape[1] > image_shape[0]:
            new_width = int(size_goal)
            new_height = int(image_shape[0] * new_width / image_shape[1])
        else:
            new_height = int(size_goal)
            new_width = int(image_shape[1] * new_height / image_shape[0])

        image_resized = image.resize((new_width, new_height))

        FileManager.close(image)

        return image_resized

    def squareImage(self, file, size_goal):
        resizedImage = self.resizeImage(file, size_goal)

        squareImage = Image.new("RGB", (size_goal, size_goal), "white")
        squareImage.paste(resizedImage, (0, 0))
        resizedImage.close()

        return squareImage

    def saveImage(self, image, destination, mode):
        return FileManager.save(image, destination, mode)

    def deleteImage(self, filepath):
        FileManager.delete(filepath)

#to do:
    # generate square images (save them somewhere or no?)
    # scale down RGB values (divide by 255)
    # reduce dimensionality from 4 to 2
    # store square images as csv files
    # be able to delete old images / csv files at location
    # separate csv file generation from machine learning (for time reduction purposes)

#in the future:
    #make file manager a class object
    #get around using imageSquarer as a mask