import unittest
import ImageSquarer
import FileManager
import numpy as np


class ImageSquarerTestCase(unittest.TestCase):
    """Base class for all ImageSquarer tests."""

    def assertResizeEqual(self, imageSquarer, file, newSize, shape):
        image = np.array(imageSquarer.resizeImage(file, newSize))
        self.assertEqual(image.shape, shape)


class ImageSquarerTest(ImageSquarerTestCase):
    def setUp(self):
        self.testFile1 = 'resources/cat.4.jpg'
        self.testFile2 = 'resources/cat.2.jpg'

    def testResizeHorizontalImage(self):
        imageSquarer = ImageSquarer.ImageSquarer()
        r = FileManager.read(self.testFile1)
        shape = np.array(r).shape
        FileManager.close(r)

        width = 256
        height = int(shape[0] * width / shape[1])
        new_shape = (height, width, 3)

        self.assertResizeEqual(imageSquarer, self.testFile1, 256, new_shape)

    def testResizeVerticalImage(self):
        imageSquarer = ImageSquarer.ImageSquarer()
        r = FileManager.read(self.testFile2)
        shape = np.array(r).shape
        FileManager.close(r)

        height = 256
        width = int(shape[1] * height / shape[0])
        new_shape = (height, width, 3)

        self.assertResizeEqual(imageSquarer, self.testFile2, 256, new_shape)

    def testSquarifyImage(self):
        imageSquarer = ImageSquarer.ImageSquarer()
        size_goal = 256
        image = np.array(imageSquarer.squareImage(self.testFile1, size_goal))
        self.assertEqual(image.shape, (size_goal, size_goal, 3))