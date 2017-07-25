from coccimorph.segment import binaryze
import numpy as np
import unittest


class TestMethods(unittest.TestCase):
    def test_binaryze(self):
        im = np.array([[[0, 0, 0], [0, 10, 20]], [[20, 20, 20], [10, 30, 50]]])
        imbin = binaryze(im, 10)
        self.assertEqual(255, imbin[0][0])
        self.assertEqual(0, imbin[0][1])
        self.assertEqual(0, imbin[1][0])
        self.assertEqual(0, imbin[1][1])

if __name__ == '__main__':
    unittest.main()
