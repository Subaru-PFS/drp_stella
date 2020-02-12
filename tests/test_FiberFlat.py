import numpy as np

import lsst.utils.tests
import lsst.afw.image as afwImage
from pfs.drp.stella.constructFiberFlatTask import ConstructFiberFlatTask


class FiberFlatTestCase(lsst.utils.tests.TestCase):
    def testCorrectLowVariance(self):
        """ Tests the ConstructFiberFlatTask.correctLowVarianceImage method
        """
        width = 100
        height = 50

        defaultVarValue = 10.0
        defaultImageValue = 1.0
        defaultMaskValue = 0x0
        minVariance = 4.0

        affectedPixelRow = 23
        affectedPixelCol = 56

        maskedImage = afwImage.MaskedImageF(width, height)  # Image for extraction
        maskedImage.image.array[:] = defaultImageValue
        maskedImage.mask.array[:] = defaultMaskValue
        maskedImage.variance.array[:] = defaultVarValue
        maskedImage.variance.array[affectedPixelRow, affectedPixelCol] = defaultImageValue

        cfft = ConstructFiberFlatTask()
        cfft.correctLowVarianceImage(maskedImage, minVariance)

        imgArr = maskedImage.getImage().getArray()
        varArr = maskedImage.getVariance().getArray()
        maskArr = maskedImage.getMask().getArray()
        isLowVar = varArr < minVariance

        self.assertFalse(np.any(isLowVar))
        self.assertFloatsEqual(imgArr, defaultImageValue)
        bitMask = maskedImage.getMask().getPlaneBitMask(['NO_DATA'])
        self.assertFloatsEqual(maskArr[affectedPixelRow, affectedPixelCol], bitMask)
        isNotAffected = (maskArr == defaultMaskValue)
        self.assertFloatsEqual(varArr[isNotAffected], defaultVarValue)


def setup_module(module):
    lsst.utils.tests.init()


if __name__ == "__main__":
    setup_module(sys.modules["__main__"])
    unittest.main(failfast=True, argv=[__file__] + argv)
