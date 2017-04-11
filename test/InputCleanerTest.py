import unittest

from PIL import Image
import numpy as np

import InputCleaner


class InputCleanerTest(unittest.TestCase):

    def setUp(self):

        self.InputCleaner = InputCleaner.InputCleaner()

    def test_image_to_array(self):
        image = Image.new('RGB', (2, 2), "white")
        actual = self.InputCleaner.image_to_array(image)

        self.assertEqual(actual.size, 12)

    def testScaleWhiteImage(self):

        image = Image.new('RGB', (8, 8), "white")
        image = np.array(image)
        scale = 255

        actual = self.InputCleaner.scale_image(image, scale)
        self.assertEqual(np.amax(actual), 1)

    def testScaleBlackImage(self):

        image = Image.new('RGB', (8, 8), "black")
        image = np.array(image)
        scale = 255

        actual = self.InputCleaner.scale_image(image, scale)
        self.assertEqual(np.amax(actual), 0)

    def testCondenseSmallArray(self):

        image = np.array([[1, 3], [2, 4]])

        actual = self.InputCleaner.condense(image)

        e = np.array(([1, 3, 2, 4]))
        e = e.reshape((1, e.size))

        self.assertEqual(actual.shape, e.shape)
        self.assertTrue(np.array_equal(actual, e))

    def testCondenseLargerArray(self):

        image = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        actual = self.InputCleaner.condense(image)

        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8])
        expected = expected.reshape((1, expected.size))

        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.array_equal(actual, expected))

    def testAddToArray(self):

        array = np.array([3, 4])
        array = array.reshape((1, array.size))

        image = np.array([1, 2])
        image = image.reshape((1, image.size))

        actual = self.InputCleaner.add_to_array(array, image)

        expected = np.array([[3, 4], [1, 2]])

        self.assertEqual(actual.shape, expected.shape)
        self.assertTrue(np.array_equal(expected, actual))

    def testAddToArrayVerticalImage(self):

        pass