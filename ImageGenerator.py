import os

from ImageSquarer import ImageSquarer


class ImageGenerator:
    def __init__(self):
        self.imageSquarer = ImageSquarer()

    def generateSquareImages(self, path, size_goal, destination):
        self.clearDirectory(destination)

        images_saved = 0

        for filename in os.listdir(path):
            if '.jpg' in filename:
                print("generating {}".format(filename))
                dest = path + '/' + filename
                new_image = self.imageSquarer.squareImage(dest, size_goal)
                self.imageSquarer.saveImage(new_image, destination + '/' + filename, "jpeg")
                images_saved += 1

        return images_saved

    def clearDirectory(self, path):
        for file in os.listdir(path):
            print("removing {}".format(file))
            filepath = os.path.join(path, file)
            self.imageSquarer.deleteImage(filepath)


ig = ImageGenerator()
ig.generateSquareImages('data/train', 50, 'data/train/modified')