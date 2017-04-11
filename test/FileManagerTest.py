from unittest import mock
from PIL import Image as Image

import FileManager
import unittest


class FileManagerTest(unittest.TestCase):

    def setUp(self):
        self.testFile = 'resources/cat.4.jpg'

    def testReadAndCloseFile(self):
        im = FileManager.read(self.testFile)
        assert im is not None
        assert FileManager.close(im) is True

    @mock.patch('FileManager.Image')
    def testSaveFileKeyError(self, mock_image):
        image = Image.new("RGB", (256, 256), "white")

        #with self.assertRaises(KeyError):
        result = FileManager.save(image, None, "jpg")

        self.assertFalse(result)

    @mock.patch('FileManager.Image.Image.save')
    def testSaveFileLowerMode(self, mock_save):
        image = Image.new("RGB", (256,256), "white")

        result = FileManager.save(image, None, "jpeg")

        self.assertTrue(result)
        args, kwargs = mock_save.call_args_list[0]
        self.assertEquals(args, (None, "JPEG"))
