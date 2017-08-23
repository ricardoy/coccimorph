from coccimorph.segment import binaryze, Segmentator
import numpy as np
import os
import pandas as pd
import unittest


class TestMethods(unittest.TestCase):
    # def test_binaryze(self):
    #     im = np.array([[[0, 0, 0], [0, 10, 20]], [[20, 20, 20], [10, 30, 50]]])
    #     imbin = binaryze(im, 10)
    #     self.assertEqual(255, imbin[0][0])
    #     self.assertEqual(0, imbin[0][1])
    #     self.assertEqual(0, imbin[1][0])
    #     self.assertEqual(0, imbin[1][1])

    def test_contour(self):
        basedir = os.path.dirname(__file__) + '/../data'
        img_filename = '%s/%s' % (basedir, 'ACE104.bmp')
        seg = Segmentator(img_filename, 140, None)
        seg.process_contour()
        v = []
        for a, b in zip(seg.vx, seg.vy):
            v.append((a, b))
        v = np.array(v)

        countor_filename = '%s/%s' % (basedir, 'countor.csv')
        df  = pd.read_csv(countor_filename, header=-1)
        y = df.as_matrix()
        self.assertTrue(np.array_equal(v, y))


if __name__ == '__main__':
    unittest.main()
