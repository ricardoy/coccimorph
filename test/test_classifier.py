from coccimorph.content import rabbit_species
from coccimorph.content import fowl_species
from coccimorph.content import generate_probability_classifier_rabbit
from coccimorph.content import generate_probability_classifier_fowl
from coccimorph.content import generate_similarity_classifier_fowl
from coccimorph.content import generate_similarity_classifier_rabbit
from coccimorph.content import FeatureExtractor
from coccimorph.segment import Segmentator
from coccimorph.functions import fftderiv, entropy
import numpy as np
import os
import pandas as pd
import unittest

rabbit_map = dict()
for i, s in enumerate(rabbit_species):
    rabbit_map[s] = i

fowl_map = dict()
for i, s in enumerate(fowl_species):
    fowl_map[s] = i


class TestMethods(unittest.TestCase):
    def test_rabbit_probability_classifier(self):
        xvector = [
            6.700e-03,
            6.867e-03,
            4.082e+02,
            361,
            210,
            2.758e-02,
            2.109e-02,
            62043,
            1.061e+01,
            5.191e-04,
            2.633e+02,
            1.668e-01,
            3.335e+01
        ]
        c = generate_probability_classifier_rabbit()
        c.classify(xvector)
        self.assertAlmostEqual(52.95, c.taxa_acerto[rabbit_map['E. coecicola']], delta=.01)
        self.assertAlmostEqual(23.42, c.taxa_acerto[rabbit_map['E. media']], delta=.01)
        self.assertAlmostEqual(18.91, c.taxa_acerto[rabbit_map['E. vejdovskyi']], delta=.01)
        self.assertAlmostEqual(2.0, c.taxa_acerto[rabbit_map['E. flavescens']], delta=.01)
        self.assertAlmostEqual(1.41, c.taxa_acerto[rabbit_map['E. piriformis']], delta=.01)
        self.assertAlmostEqual(1.31, c.taxa_acerto[rabbit_map['E. intestinalis']], delta=.01)
        self.assertAlmostEqual(0.0, c.taxa_acerto[rabbit_map['E. magna']], delta=.01)

    def test_fowl_probability_classifier(self):
        xvector = [
            1.124e-02,
            4.228e-03,
            3.491e+02,
            204,
            147,
            1.243e-02,
            5.725e-02,
            23879,
            1.213e+01,
            2.466e-04,
            4.045e+02,
            1.080e-01,
            3.827e+01
        ]
        c = generate_probability_classifier_fowl()
        c.classify(xvector)
        self.assertAlmostEqual(99.30, c.taxa_acerto[fowl_map['E. acervulina']], delta=.01)
        self.assertAlmostEqual(0.70, c.taxa_acerto[fowl_map['E. mitis']], delta=.01)

    def test_rabbit_similarity_classifier(self):
        xvector = [
            0.0129782,
            0.00211385,
            324.234,
            159.316,
            151.269,
            0.0144904,
            0.0173005,
            18827,
            11.9328,
            0.000324872,
            463.133,
            0.130554,
            36.2551
        ]

        c = generate_similarity_classifier_rabbit()
        taxa_acerto = c.classify(xvector)
        self.assertAlmostEqual(67.35, taxa_acerto[rabbit_map['E. exigua']], delta=.01)

    def test_fowl_similarity_classifier(self):
        xvector = [
            0.0112434,
            0.00422792,
            349.083,
            204.885,
            147.666,
            0.0124258,
            0.0572524,
            23879,
            12.1338,
            0.000246645,
            404.536,
            0.107968,
            38.2667
        ]

        c = generate_similarity_classifier_fowl()
        taxa_acerto = c.classify(xvector)
        self.assertAlmostEqual(76.80, taxa_acerto[fowl_map['E. acervulina']], delta=.01)
        self.assertAlmostEqual(14.87, taxa_acerto[fowl_map['E. mitis']], delta=.01)
        self.assertAlmostEqual(13.95, taxa_acerto[fowl_map['E. necatrix']], delta=.01)

    def test_fowl_similarity_classifier_after_gray_image(self):
        basedir = os.path.dirname(__file__) + '/data'
        img_filename = '%s/%s' % (basedir, 'ACE104.bmp')
        seg = Segmentator(img_filename, 140, None)
        seg.process_contour()

        f1 = seg.vx
        f2 = seg.vy

        n = len(f1)
        sigma = 10

        d1x = fftderiv(f1, 1, sigma, n)
        d2x = fftderiv(f1, 2, sigma, n)
        d1y = fftderiv(f2, 1, sigma, n)
        d2y = fftderiv(f2, 2, sigma, n)

        k = (d1x * d2y - d1y * d2x) / np.power(np.power(d1x, 2) + np.power(d1y, 2), 1.5)

        xvector = []
        xvector.append(np.average(k))
        xvector.append(np.std(k))
        xvector.append(entropy(k))

        feature_extractor = FeatureExtractor(img_filename, None)
        gray_image = pd.read_csv('%s/%s' % (basedir, 'gray_image.csv'), sep=' ', header=None).as_matrix()
        feature_extractor.img_gray = gray_image

        feature_extractor.content_read(f1, f2, n)
        feature_extractor.set_co_matrix(2)
        feature_extractor.eigens()

        xvector.append(feature_extractor.high_diameter)
        xvector.append(feature_extractor.less_diameter)
        xvector.append(feature_extractor.sym_high_pc)
        xvector.append(feature_extractor.sym_less_pc)
        xvector.append(feature_extractor.obj_size)
        xvector.append(feature_extractor.obj_entropy)
        xvector.append(feature_extractor.mcc_asm())
        xvector.append(feature_extractor.mcc_con())
        xvector.append(feature_extractor.mcc_idf())
        xvector.append(feature_extractor.mcc_ent())

        c = generate_similarity_classifier_fowl()
        taxa_acerto = c.classify(xvector)
        self.assertAlmostEqual(76.80, taxa_acerto[fowl_map['E. acervulina']], delta=.01)
        self.assertAlmostEqual(14.87, taxa_acerto[fowl_map['E. mitis']], delta=.01)
        self.assertAlmostEqual(13.95, taxa_acerto[fowl_map['E. necatrix']], delta=.01)

if __name__ == '__main__':
    unittest.main()
