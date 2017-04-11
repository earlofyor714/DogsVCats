from unittest import mock

import ImageGenerator
import unittest


class ImageGeneratorTest(unittest.TestCase):

    @mock.patch('ImageGenerator.os')
    @mock.patch('ImageGenerator.ImageSquarer.squareImage')
    @mock.patch('ImageGenerator.ImageSquarer.saveImage')
    def testGenerateImageIterator(self, mock_saveImage, mock_ImageSquarer, mock_os):
        mock_os.listdir.return_value = ['a.jpg']

        imageGenerator = ImageGenerator.ImageGenerator()
        size_goal = 256

        result = imageGenerator.generateSquareImages('resources', size_goal, 'b')
        self.assertEqual(result, 1)
        assert mock_ImageSquarer.call_count == 1
        assert mock_saveImage.call_count == 1

    @mock.patch('ImageGenerator.os')
    @mock.patch('ImageGenerator.ImageSquarer.squareImage')
    @mock.patch('ImageGenerator.ImageSquarer.saveImage')
    def testGenerateImageNoJpg(self, mock_saveImage, mock_ImageSquarer, mock_os):
        mock_os.listdir.return_value = ['b']
        size_goal = 256
        imageGenerator = ImageGenerator.ImageGenerator()

        result = imageGenerator.generateSquareImages('resources', size_goal, 'b')
        self.assertEqual(result, 0)
        assert not mock_ImageSquarer.called
        assert not mock_saveImage.called

    @mock.patch('ImageGenerator.os')
    @mock.patch('ImageGenerator.ImageSquarer.squareImage')
    @mock.patch('ImageGenerator.ImageSquarer.saveImage')
    def testImageNameCorrect(self, mock_saveImage, mock_squareImage, mock_os):
        mock_os.listdir.return_value = ['a.jpg', 'b.jpg']
        size_goal = 256

        mock_squareImage.return_value = "image"

        imageGenerator = ImageGenerator.ImageGenerator()
        result = imageGenerator.generateSquareImages('resources', size_goal, 'c')
        self.assertEqual(result, 2)

        args, kwargs = mock_squareImage.call_args_list[0]
        self.assertEquals(args, ('resources/a.jpg', size_goal))

        args, kwargs = mock_squareImage.call_args_list[1]
        self.assertEquals(args, ('resources/b.jpg', size_goal))

        args, kwargs = mock_saveImage.call_args_list[0]
        self.assertEquals(args, ('image', 'c/a.jpg', "jpg"))

        args, kwargs = mock_saveImage.call_args_list[1]
        self.assertEquals(args, ('image', 'c/b.jpg', "jpg"))

# python -m unittest ImageGeneratorTest.py
# coverage report -m

#test iterates through all images in a folder
#test iterate on empty folder
#test creates correct number of images
#test names of new images are correct