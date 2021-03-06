import argparse
import cv2
import numpy as np
from coccimorph.aux import load_image


class Segmentator(object):
    def __init__(self, filename, threshold, scale):
        self.img = load_image(filename, scale)
        self.img_bin = binaryze(self.img, threshold)
        self.height, self.width, _ = self.img.shape
        self.vx = []
        self.vy = []
        self.checkpoint = 0
        self.invert = [
            [0, 4],
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 0],
            [5, 1],
            [6, 2],
            [7, 3]
        ]

    def save_segmentation(self, filename):
        img = np.copy(self.img)
        for x, y in zip(self.vx, self.vy):
            img[x, y] = 255
        cv2.imwrite(filename, np.transpose(img, axes=[1, 0, 2]))
        print('File %s was saved.'%(filename))

    def process_contour(self):
        fim = False
        starting_point_found = False
        self.checkpoint = 0

        i = 0
        while i < self.height and not fim:
            j = 0
            while j < self.width and not fim:
                if self.img_bin[i][j] == 255:
                    self.vx.append(i)
                    self.vy.append(j-1)
                    starting_point_found = True
                    fim = True
                j += 1
            i += 1

        if not starting_point_found:
            self.vx.append(0)
            self.vy.append(0)

        if self.vx[0] > 1 and self.vy[0] > 1 and \
                self.vx[0] < self.height - 1 and self.vy[0] < self.width - 1:
            n = 2

            x4 = self.vx[0]
            y4 = self.vy[0] - 1
            x5 = self.vx[0] + 1
            y5 = self.vy[0] - 1
            x6 = self.vx[0] + 1
            y6 = self.vy[0]
            x7 = self.vx[0] + 1
            y7 = self.vy[0] + 1
            x0 = self.vx[0]
            y0 = self.vy[0] + 1
            dcn = 0

            next_pixel = (0, 0)
            if self.img_bin[x4, y4] == 0 and self.img_bin[x5, y5] == 255:
                next_pixel = (x4, y4)
                dcn = 4
            elif self.img_bin[x5, y5] == 0 and self.img_bin[x6, y6] == 255:
                next_pixel = (x5, y5)
                dcn = 5
            elif self.img_bin[x6, y6] == 0 and self.img_bin[x7, y7] == 255:
                next_pixel = (x6, y6)
                dcn = 6
            elif self.img_bin[x7, y7] == 0 and self.img_bin[x0, y0] == 255:
                next_pixel = (x7, y7)
                dcn = 7

            while not(next_pixel[0] == self.vx[0] and next_pixel[1] == self.vy[0]):
                self.vx.append(int(next_pixel[0]))
                self.vy.append(int(next_pixel[1]))
                dpc = dcn

                # w_vect = (next_pixel[0], next_pixel[1], dcn)
                retvals = self._find_next(self.vx[-1], self.vy[-1], dpc)
                next_pixel = (retvals[0], retvals[1])
                dcn = retvals[2]
                n += 1

                if n < 20:
                    i = 0
                    while(i < n-1):
                        if next_pixel[0] == self.vx[i] and \
                                        next_pixel[1] == self.vy[i] and i > 0:
                            next_pixel = (self.vx[0], self.vy[0])
                            n -= 1
                            self.checkpoint = 1
                        i += 1

    def _find_next(self, pcx: int, pcy: int, dpc: int):
        w2 = np.zeros(3, dtype=np.int)
        dcp = self.invert[dpc][1]
        for r in range(7):
            dE = (dcp + r) % 8
            dI = (dcp + r + 1) % 8
            pe = self.chain_point(pcx, pcy, dE)
            pi = self.chain_point(pcx, pcy, dI)
            if self.is_background(pe) and self.is_object(pi):
                w2[0] = pe[0]
                w2[1] = pe[1]
                w2[2] = dE
        return w2

    def is_background(self, pe):
        return self.img_bin[pe[0], pe[1]] == 0

    def is_object(self, pi):
        return self.img_bin[pi[0], pi[1]] == 255

    def chain_point(self, pcx, pcy, d):
        if d == 0:
            return pcx, pcy + 1
        elif d == 1:
            return pcx - 1, pcy + 1
        elif d == 2:
            return pcx - 1, pcy
        elif d == 3:
            return pcx - 1, pcy - 1
        elif d == 4:
            return pcx, pcy - 1
        elif d == 5:
            return pcx + 1, pcy - 1
        elif d == 6:
            return pcx + 1, pcy
        elif d == 7:
            return pcx + 1, pcy + 1
        else:
            raise ValueError('Parameter d should be an integer in [0, 7].')


def binaryze(img, threshold):
    img_grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binarizer = np.vectorize(lambda x: 255 if x < threshold else 0)
    img_bin = binarizer(img_grayscale)
    # print("primeiro pixel", img_grayscale[0, 0], "cores:", img[0, 0])
    return img_bin


def segment(filename, threshold, binary_file, segmented_file, scale=None):
    seg = Segmentator(filename, threshold, scale)
    seg.process_contour()
    if binary_file is None:
        binary_file = '/tmp/binary.png'
    cv2.imwrite(binary_file, np.transpose(abs(seg.img_bin - 255), axes=[1, 0]))
    print('File %s was saved.'%(binary_file))
    if segmented_file is None:
        segmented_file = '/tmp/segmented.png'
    seg.save_segmentation(segmented_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segment image.')
    parser.add_argument('-input-file', type=str)
    parser.add_argument('-threshold', type=int)
    parser.add_argument('-scale', type=float)
    parser.add_argument('-output-binary', type=str)
    parser.add_argument('-output-segmented', type=str)
    args = parser.parse_args()
    if args.input_file is not None and args.threshold is not None:
        segment(args.input_file,
                args.threshold,
                args.output_binary,
                args.output_segmented,
                args.scale)
