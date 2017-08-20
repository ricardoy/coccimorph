from coccimorph.segment import Segmentator
from coccimorph.functions import fftderiv, entropy
from coccimorph.content import FeatureExtractor, ClassificaGauss, ClassificaProb
import argparse
import numpy as np


def printf(label, num):
    print('%s %.3e'%(label, num))

def output_xvector(xvector):
    print()
    printf('Mean of curvature:', xvector[0])
    printf('Standard deviation from curvature:', xvector[1])
    printf('Entropy of curvature:', xvector[2])
    printf('Largest diameter:', xvector[3])
    printf('Smallest diameter:', xvector[4])
    printf('Symmetry based on first principal component:', xvector[5])
    printf('Symmetry based on second principal component:', xvector[6])
    printf('Total number of pixels:', xvector[7])
    printf('Entropy of image content:', xvector[8])
    printf('Angular second moment from co-occurrence matrix:', xvector[9])
    printf('Contrast from co-occurrence matrix:', xvector[10])
    printf('Inverse difference moment from co-occurrence matrix:', xvector[11])
    printf('Entropy of co-occurence matrix:', xvector[12])
    print()


def predict(filename, threshold, scale):
    seg = Segmentator(filename, threshold, scale)
    seg.process_contour()

    f1 = seg.vx
    f2 = seg.vy

    # print(f1)
    # print(f2)

    n = len(f1)
    sigma = 10

    # fft derivatives
    d1x = fftderiv(f1, 1, sigma, n)
    d2x = fftderiv(f1, 2, sigma, n)
    d1y = fftderiv(f2, 1, sigma, n)
    d2y = fftderiv(f2, 2, sigma, n)

    # cunvature K and its moments
    k = (d1x * d2y - d1y * d2x) / np.power(np.power(d1x, 2) + np.power(d1y, 2), 1.5)

    xvector = []
    xvector.append(np.average(k))
    xvector.append(np.std(k))
    xvector.append(entropy(k))

    feature_extractor = FeatureExtractor(filename, scale)
    feature_extractor.content_read(f1, f2, n)
    feature_extractor.set_co_matrix(2)
    feature_extractor.eigens()

    xvector.append(feature_extractor.high_diameter)
    xvector.append(feature_extractor.less_diameter)
    xvector.append(feature_extractor.sym_high_pc)
    xvector.append(feature_extractor.sym_less_pc)
    xvector.append(feature_extractor.obj_size) # problema
    xvector.append(feature_extractor.obj_entropy)
    xvector.append(feature_extractor.mcc_asm())
    xvector.append(feature_extractor.mcc_con()) # problema
    xvector.append(feature_extractor.mcc_idf())
    xvector.append(feature_extractor.mcc_ent())

    output_xvector(xvector)

    prob_classifier = ClassificaProb()
    prob_classifier.classify(xvector)

    classifier = ClassificaGauss()
    classifier.classify(xvector)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity classifier.')
    parser.add_argument('-input-file', type=str)
    parser.add_argument('-threshold', type=int)
    parser.add_argument('-scale', type=float)
    args = parser.parse_args()
    if args.input_file is not None and args.threshold is not None:
        predict(args.input_file, args.threshold, args.scale)
