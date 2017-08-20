from coccimorph.content import rabbit_species
from coccimorph.content import fowl_species
from coccimorph.content import generate_probability_classifier_rabbit
from coccimorph.content import generate_probability_classifier_fowl
from coccimorph.content import generate_similarity_classifier_fowl
import unittest

rabbit_map = dict()
for i, s in enumerate(rabbit_species):
    rabbit_map[s] = i

fowl_map = dict()
for i, s in enumerate(fowl_species):
    fowl_map[s] = i


class TestMethods(unittest.TestCase):
    def test_rabbit_classifier(self):
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

    def test_fowl_similarity_classifier(self):
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

        c = generate_similarity_classifier_fowl()
        taxa_acerto = c.classify(xvector)
        self.assertAlmostEqual(76.80, taxa_acerto[fowl_map['E. acervulina']], delta=.01)
        self.assertAlmostEqual(14.87, taxa_acerto[fowl_map['E. mitis']], delta=.01)
        self.assertAlmostEqual(13.95, taxa_acerto[fowl_map['E. necatrix']], delta=.01)

if __name__ == '__main__':
    unittest.main()
