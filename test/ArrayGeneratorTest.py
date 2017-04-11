import unittest
from unittest import mock

from PIL import Image

import ArrayGenerator


class BaseArrayGeneratorTest(unittest.TestCase):
    def generate_general_values(self, image_size):
        arrayGenerator = ArrayGenerator.ArrayGenerator()
        return arrayGenerator.generate_inputs_labels('', image_size)


class ArrayGeneratorTest(BaseArrayGeneratorTest):

    @mock.patch('ArrayGenerator.os')
    @mock.patch('ArrayGenerator.FileManager')
    def testGenerateArraySingleX(self, mock_file_manager, mock_os):
        image_length = 2
        image_size = image_length * image_length * 3

        mock_os.listdir.return_value = ['a.jpg']

        image = Image.new('RGB', (image_length, image_length), "white")
        mock_file_manager.read.return_value = image

        X, y = self.generate_general_values(image_size)

        self.assertEqual(X.shape, (1, image_size))

    @mock.patch('ArrayGenerator.os')
    @mock.patch('ArrayGenerator.FileManager')
    def testGenerateArrayMultiX(self, mock_file_manager, mock_os):
        image_length = 2
        image_size = image_length * image_length * 3

        mock_os.listdir.return_value = ['a.jpg', 'cat.jpg']

        image = Image.new('RGB', (image_length, image_length), "white")
        mock_file_manager.read.return_value = image

        X, y = self.generate_general_values(image_size)

        self.assertEqual(X.shape, (2, image_size))


    @mock.patch('ArrayGenerator.os')
    @mock.patch('ArrayGenerator.FileManager')
    def testGenerateArrayDogY(self, mock_file_manager, mock_os):
        image_length = 2
        image_size = image_length * image_length * 3

        mock_os.listdir.return_value = ['a.jpg']

        image = Image.new('RGB', (image_length, image_length), "white")
        mock_file_manager.read.return_value = image

        X, y = self.generate_general_values(image_size)

        self.assertEqual(y.shape, (1, 1))
        self.assertEqual(y[0], 0)

    @mock.patch('ArrayGenerator.os')
    @mock.patch('ArrayGenerator.FileManager')
    def testGenerateArrayCatY(self, mock_file_manager, mock_os):
        image_length = 2
        image_size = image_length * image_length * 3

        mock_os.listdir.return_value = ['cat.jpg']

        image = Image.new('RGB', (image_length, image_length), "white")
        mock_file_manager.read.return_value = image

        X, y = self.generate_general_values(image_size)

        self.assertEqual(y.shape, (1, 1))
        self.assertEqual(y[0], 1)

    @mock.patch('ArrayGenerator.os')
    @mock.patch('ArrayGenerator.FileManager')
    def testGenerateArrayMultiY(self, mock_file_manager, mock_os):
        image_length = 2
        image_size = image_length * image_length * 3

        mock_os.listdir.return_value = ['dog.jpg', 'cat.jpg']

        image = Image.new('RGB', (image_length, image_length), "white")
        mock_file_manager.read.return_value = image

        X, y = self.generate_general_values(image_size)

        self.assertEqual(y.shape, (2, 1))
        self.assertEqual(y[0], 0)
        self.assertEqual(y[1], 1)
