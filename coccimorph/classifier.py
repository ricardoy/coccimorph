from coccimorph.segment import Segmentator
from coccimorph.functions import fftderiv, entropy
from coccimorph.content import FeatureExtractor
from coccimorph.content import generate_similarity_classifier_fowl
from coccimorph.content import generate_probability_classifier_fowl
from coccimorph.content import generate_similarity_classifier_rabbit
from coccimorph.content import generate_probability_classifier_rabbit
import argparse
import numpy as np


def float_feature_to_string(label, num):
    return label, '%.3e' % num


def integer_feature_to_string(label, num):
    return label, int(num)


def get_feature_labels_and_values(xvector):
    s = list()
    s.append(float_feature_to_string('Mean of curvature', xvector[0]))
    s.append(float_feature_to_string('Standard deviation from curvature', xvector[1]))
    s.append(float_feature_to_string('Entropy of curvature', xvector[2]))
    s.append(integer_feature_to_string('Largest diameter', xvector[3]))
    s.append(integer_feature_to_string('Smallest diameter', xvector[4]))
    s.append(float_feature_to_string('Symmetry based on first principal component', xvector[5]))
    s.append(float_feature_to_string('Symmetry based on second principal component', xvector[6]))
    s.append(integer_feature_to_string('Total number of pixels', xvector[7]))
    s.append(float_feature_to_string('Entropy of image content', xvector[8]))
    s.append(float_feature_to_string('Angular second moment from co-occurrence matrix', xvector[9]))
    s.append(float_feature_to_string('Contrast from co-occurrence matrix', xvector[10]))
    s.append(float_feature_to_string('Inverse difference moment from co-occurrence matrix', xvector[11]))
    s.append(float_feature_to_string('Entropy of co-occurence matrix', xvector[12]))
    return s


def get_feature_labels_and_raw_values(xvector):
    s = list()
    s.append(('Mean of curvature', xvector[0]))
    s.append(('Standard deviation from curvature', xvector[1]))
    s.append(('Entropy of curvature', xvector[2]))
    s.append(('Largest diameter', xvector[3]))
    s.append(('Smallest diameter', xvector[4]))
    s.append(('Symmetry based on first principal component', xvector[5]))
    s.append(('Symmetry based on second principal component', xvector[6]))
    s.append(('Total number of pixels', xvector[7]))
    s.append(('Entropy of image content', xvector[8]))
    s.append(('Angular second moment from co-occurrence matrix', xvector[9]))
    s.append(('Contrast from co-occurrence matrix', xvector[10]))
    s.append(('Inverse difference moment from co-occurrence matrix', xvector[11]))
    s.append(('Entropy of co-occurence matrix', xvector[12]))
    return s


def output_xvector(xvector):
    print()
    for label, value in get_feature_labels_and_values(xvector):
        print('{}: {}'.format(label, value))
    print()


def predict(filename, threshold, scale, fowl, rabbit, output_data=False):
    seg = Segmentator(filename, threshold, scale)
    seg.process_contour()

    f1 = seg.vx
    f2 = seg.vy

    n = len(f1)
    sigma = 10

    # fft derivatives
    d1x = fftderiv(f1, 1, sigma, n)
    d2x = fftderiv(f1, 2, sigma, n)
    d1y = fftderiv(f2, 1, sigma, n)
    d2y = fftderiv(f2, 2, sigma, n)

    # with open('/tmp/novo', 'w') as fh:
    #     fh.write('d1x\n')
    #     for x in d1x:
    #         fh.write(str(x))
    #         fh.write('\n')

    # with open('/tmp/novo', 'w') as fh:
    #     fh.write('d2x\n')
    #     for x in d2x:
    #         fh.write(str(x))
    #         fh.write('\n')

    # curvature K and its moments
    k = (d1x * d2y - d1y * d2x) / np.power(np.power(d1x, 2) + np.power(d1y, 2), 1.5)

    # with open('/tmp/novo', 'w') as fh:
    #     fh.write('k\n')
    #     for x in k:
    #         fh.write(str(x))
    #         fh.write('\n')

    xvector = list()
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
    xvector.append(feature_extractor.obj_size)
    xvector.append(feature_extractor.obj_entropy)
    xvector.append(feature_extractor.mcc_asm())
    xvector.append(feature_extractor.mcc_con())
    xvector.append(feature_extractor.mcc_idf())
    xvector.append(feature_extractor.mcc_ent())

    if fowl:
        prob_classifier = generate_probability_classifier_fowl()
        prob_result = prob_classifier.classify(xvector)
        simi_classifier = generate_similarity_classifier_fowl()
        simi_result = simi_classifier.classify(xvector)
    if rabbit:
        prob_classifier = generate_probability_classifier_rabbit()
        prob_result = prob_classifier.classify(xvector)
        simi_classifier = generate_similarity_classifier_rabbit()
        simi_result = simi_classifier.classify(xvector)

    if output_data:
        output_xvector(xvector)
        print('\nProbability classification:')
        for label, value in prob_result.items():
            print('%s: %.4f' % (label, value))
        print('\nSimilarity classification:')
        for label, value in simi_result.items():
            print('%s: %.4f' % (label, value))
        print()

    return {
        'features': get_feature_labels_and_raw_values(xvector),
        'probability': prob_result,
        'similarity': simi_result
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Similarity classifier.')
    parser.add_argument('-input-file', type=str)
    parser.add_argument('-threshold', type=int)
    parser.add_argument('-scale', type=float)
    parser.add_argument('--fowl', action='store_true')
    parser.add_argument('--rabbit', action='store_true')
    args = parser.parse_args()
    if not args.fowl and not args.rabbit:
        parser.print_help()
        exit(-1)
    if args.input_file is not None and args.threshold is not None:
        predict(args.input_file, args.threshold, args.scale, args.fowl, args.rabbit, True)
