import os
from pytesseract import pytesseract
import image_enchant

os.environ['TESSDATA_PREFIX'] = './tesseract-mrz'


class Back:
    def __init__(self, img):
        self._image = img
        self.set_MRZ_image()
        self.set_MRZ()

    def set_MRZ_image(self):
        m = image_enchant.get_mrz_image(self._image)
        self._MRZ_image = image_enchant.convert_to_binary(m)

    # https://github.com/DoubangoTelecom/tesseractMRZ/
    def set_MRZ(self):
        self.MRZ = pytesseract.image_to_string(
            self._MRZ_image, lang='mrz', config='--psm 6')
