import math
import numpy as np
import os
import pandas as pd
from coccimorph.aux import load_image
import cv2


RED = (0, 0, 255)

fowl_species = [
    'E. acervulina',
    'E. maxima',
    'E. brunetti',
    'E. mitis',
    'E. praecox',
    'E. tenella',
    'E. necatrix'
]

rabbit_species = [
    'E. coecicola',
    'E. exigua',
    'E. flavescens',
    'E. intestinalis',
    'E. irresidua',
    'E. magna',
    'E. media',
    'E. perforans',
    'E. piriformis',
    'E. stiedai',
    'E. vejdovskyi'
]

basedir = os.path.dirname(__file__) + '/../prototypes'


def dilate(ima):
    '''
    Morphological dilation of binary matrix ima using
    as default the structuring element(SE)
    [0 1 0
     1 1 1
     0 1 0]
    :param ima: a binary array
    :return:
    '''
    dx, dy = ima.shape
    se = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    ima_temp = np.zeros((dx, 500), dtype=np.int)
    for m in range(dx):
        for n in range(dy):
            ima_temp[m, n] = ima[m, n]

    for m in range(1, dx - 1):
        for n in range(1, dy - 1):
            if ima_temp[m, n] == 1:
                for i in range(3):
                    for j in range(3):
                        mw = m - 1
                        nw = n - 1
                        if ima[mw + i, nw + j] == 0:
                            ima[mw + i, nw + j] = ima[mw + i, nw + j] or se[i][j]
    return ima


def erode(ima: np.ndarray):
    dx, dy = ima.shape
    ima_temp = np.zeros((dx, 500), dtype=np.int)

    for m in range(dx):
        for n in range(dy):
            ima_temp[m, n] = ima[m, n]

    for m in range(1, dx - 1):
        for n in range(1, dy - 1):
            if ima_temp[m, n] == 1:
                aux = 1
                aux *= ima_temp[m, n]
                aux *= ima_temp[m - 1, n]
                aux *= ima_temp[m + 1, n]
                aux *= ima_temp[m, n - 1]
                aux *= ima_temp[m, n + 1]
                ima[m, n] = aux

    for i in range(dx):
        ima[i, 0] = 0
        ima[i, dy - 1] = 0

    for i in range(dy):
        ima[0, i] = 0
        ima[dx - 1, i] = 0

    return ima

class FeatureExtractor:
    def __init__(self, filename, scale):
        self.img = load_image(filename, scale)
        self.height, self.width, _ = self.img.shape
        self.img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.ima = np.zeros((self.height, self.width), dtype=np.int)
        self.vx = []
        self.vy = []
        self.wEnt = None

        self.obj_entropy = 0.0
        self.obj_size = 0.0

        self.mcc = None

    def set_co_matrix(self, d: int):
        aux_mcc = np.zeros((256, 256), dtype=np.int)
        ro = 0
        for x in range(self.height):
            for y in range(self.width-d):
                if self.ima[x, y] > 0 and self.ima[x, y + d] > 0:
                    # if aux_mcc[self.ima[x, y], self.ima[x, y + d]] > 0:
                    #     print(self.ima[x, y], self.ima[x, y + d], aux_mcc[self.ima[x, y], self.ima[x, y + d]])
                    aux_mcc[self.ima[x, y], self.ima[x, y + d]] += 1
                    ro += 1

        for x in range(self.height):
            y = self.width-1
            while y > d - 1:
                if self.ima[x, y] > 0 and self.ima[x, y - d] > 0:
                    # if aux_mcc[self.ima[x, y], self.ima[x, y - d]] > 0:
                    #     print(self.ima[x, y], self.ima[x, y - d], aux_mcc[self.ima[x, y], self.ima[x, y - d]])
                    aux_mcc[self.ima[x, y], self.ima[x, y - d]] += 1
                    ro += 1
                y -= 1

        # print('ro', ro)

        # self.ima.tofile('/tmp/ima_novo')

        self.mcc = aux_mcc / float(ro)

        # with open('/tmp/mcc_novo', 'w') as fh:
        #     for i in range(255):
        #
        #         for j in range(255):
        #             fh.write('%.14f ' % (self.mcc[i][j]))
        #
        #         fh.write('%.14f' % (self.mcc[i][255]))
        #         fh.write('\n')
        #
        # print('soma total mcc', np.sum(aux_mcc), np.std(self.mcc) ** 2)

    def mcc_asm(self):
        return np.sum(np.power(self.mcc, 2))

    def mcc_con(self):
        sm = 0.0
        for i in range(256):
            for j in range(256):
                sm += self.mcc[i, j]*(i-j)*(i-j)
        return sm

    def mcc_idf(self):
        sm = 0.0
        for i in range(256):
            for j in range(256):
                sm += self.mcc[i, j] / float(1 + (i-j)*(i-j))
        return sm

    def mcc_ent(self):
        sm = 0.0
        for i in range(256):
            for j in range(256):
                if self.mcc[i, j] > 0:
                    sm += self.mcc[i, j]*np.log(self.mcc[i, j])
        return sm * sm / 2.
        # sm = np.sum(self.mcc * np.log(self.mcc))
        # return sm * sm / 2.0

    def eigens(self):
        c = np.zeros(4, dtype=np.float)

        mean_x = np.average(self.vx)
        mean_y = np.average(self.vy)

        sum0 = 0.
        sum1 = 0.
        sum2 = 0.
        sum3 = 0.
        for i in range(len(self.vx)):
            sum0 += (self.vx[i] - mean_x) * (self.vx[i] - mean_x)
            sum1 += (self.vx[i] - mean_x) * (self.vy[i] - mean_y)
            sum2 += (self.vy[i] - mean_y) * (self.vx[i] - mean_x)
            sum3 += (self.vy[i] - mean_y) * (self.vy[i] - mean_y)

        n = len(self.vx)
        c[0] = sum0/n
        c[1] = sum1/n
        c[2] = sum2/n
        c[3] = sum3/n

        k = np.reshape(c, (-1, 2))
        # print('k', k)

        # compute eigen vectors and eigen values
        eigenvalues, eigenvectors = np.linalg.eigh(k)
        # print('autovalores', eigenvalues)
        #
        # print('eigenvectors\n', eigenvectors)

        evec_inv = np.linalg.inv(eigenvectors)

        # transform to new space using inverse matrix of eigen vectors
        vx1 = np.zeros(n, dtype=np.float)
        vy1 = np.zeros(n, dtype=np.float)

        # print('inversa: ', evec_inv)

        for i in range(n):
            vx_w = evec_inv[0, 0] * self.vx[i] + evec_inv[0, 1] * self.vy[i]
            vy_w = evec_inv[1, 0] * self.vx[i] + evec_inv[1, 1] * self.vy[i]
            vx1[i] = vx_w
            vy1[i] = vy_w

        # vx1 = -1 * vx1
        # vy1 = -1 * vy1
        # with open('/tmp/novo', 'w') as fh:
        #     fh.write('valor de vx1\n')
        #     for blah in vx1:
        #         fh.write(str(blah))
        #         fh.write('\n')
        # exit()

        meanvx1 = np.average(vx1)
        meanvy1 = np.average(vy1)

        vx1 = vx1 - meanvx1
        vy1 = vy1 - meanvy1
        vx2 = np.copy(vx1)
        vy2 = np.copy(vy1)

        # searching for diameters
        # highX = np.max(vx1)
        # lessX = np.min(vx1)
        # highY = np.max(vy1)
        # lessY = np.min(vy1)
        highX = float('-Inf')
        lessX = float('Inf')
        highY = float('-Inf')
        lessY = float('Inf')
        for i in range(len(self.vx)):
            if int(vx1[i]) == 0 and vy1[i] > highY:
                highY = vy1[i]
            if int(vx1[i]) == 0 and vy1[i] < lessY:
                lessY = vy1[i]
            if int(vy1[i]) == 0 and vx1[i] > highX:
                highX = vx1[i]
            if int(vy1[i]) == 0 and vx1[i] < lessX:
                lessX = vx1[i]

        # print('meanvx1', meanvx1, 'meanvy1', meanvy1)
        # print('highX', highX, 'lessX', lessX)
        # print('highY', highY, 'lessY', lessY)
        # print('high diameter', (highY - lessY + 1))

        self.high_diameter = highY - lessY + 1
        self.less_diameter = highX - lessX + 1

        # reflects accoding to principal components
        if np.abs(int(eigenvalues[0])) > np.abs(int(eigenvalues[1])):
            for i in range(n):
                vy1[i] = -1. * vy1[i]
                vx2[i] = -1. * vx2[i]
        else:
            for i in range(n):
                vx1[i] = -1. * vx1[i]
                vy2[i] = -1. * vy2[i]

        # translate to original localization
        vx1 = vx1 + meanvx1
        vy1 = vy1 + meanvy1
        vx2 = vx2 + meanvx1
        vy2 = vy2 + meanvy1

        # return to original base
        for i in range(n):
            vx_w = eigenvectors[0,0]*vx1[i] + eigenvectors[0,1]*vy1[i]
            vy_w = eigenvectors[1,0]*vx1[i] + eigenvectors[1,1]*vy1[i]
            vx1[i] = vx_w
            vy1[i] = vy_w

            vx_w = eigenvectors[0,0]*vx2[i] + eigenvectors[0,1]*vy2[i]
            vy_w = eigenvectors[1,0]*vx2[i] + eigenvectors[1,1]*vy2[i]
            vx2[i] = vx_w
            vy2[i] = vy_w

        # compute the symmetry
        highX1 = float('-Inf')
        highY1 = float('-Inf')
        highX2 = float('-Inf')
        highY2 = float('-Inf')
        for i in range(len(self.vx)):
            if int(round(vx1[i])) > highX1:
                highX1 = int(round(vx1[i]))
            if int(round(vy1[i])) > highY1:
                highY1 = int(round(vy1[i]))
            if int(round(vx2[i])) > highX2:
                highX2 = int(round(vx2[i]))
            if int(round(vy2[i])) > highY2:
                highY2 = int(round(vy2[i]))
        """        
        TODO: original program was +3... this and the 500 columns look like
        hard constraints over the image size 
        """
        highX1 += 3
        highY1 += 3
        highX2 += 3
        highY2 += 3

        # create temporal matrices to compute erosion, dilation and rate symmetry
        ima3a = np.zeros((highX1, highY1))
        ima3b = np.zeros((highX2, highY2))

        try:
            assert (np.max(self.vx) < highX1)
        except AssertionError:
            print('Constraint for max(vx) < highX1 does not hold!')
            print(np.max(self.vx), highX1)

        try:
            assert (np.max(self.vx) < highX2)
        except AssertionError as e:
            print('Constraint for max(vx) < highX2 does not hold!')
            print(np.max(self.vx), highX2)

        #TODO write a better bound for the images dimensions
        ima2a = np.zeros((highX1*2, 500), dtype=np.int)
        ima2b = np.zeros((highX2*2, 500), dtype=np.int)
        ima4a = np.zeros((highX1*2, 500), dtype=np.int)
        ima4b = np.zeros((highX2*2, 500), dtype=np.int)

        for i in range(n):
            ima2a[int(self.vx[i]), int(self.vy[i])] = 1
            ima2b[int(self.vx[i]), int(self.vy[i])] = 1

            ima3a[int(np.round(vx1[i])), int(np.round(vy1[i]))] = 1
            ima3b[int(np.round(vx2[i])), int(np.round(vy2[i]))] = 1

        ima3a = dilate(ima3a)
        ima3a = erode(ima3a)
        for i in range(highX1):
            for j in range(highY1):
                ima4a[i, j] = ima2a[i, j] + ima3a[i, j]

        ima3b = dilate(ima3b)
        ima3b = erode(ima3b)
        for i in range(highX2):
            for j in range(highY2):
                ima4b[i, j] = ima2b[i, j] + ima3b[i, j]

        # compute symmetry index for high principal component
        sa_one = 0
        sa_two = 0
        for i in range(highX1):
            for j in range(highY1):
                if ima4a[i, j] == 1:
                    sa_one += 1
                if ima4a[i, j] == 2:
                    sa_two += 1

        self.sym_high_pc = float(sa_one) / sa_two

        # compute symmetry index for less principal component
        sa_one = 0
        sa_two = 0
        for i in range(highX2):
            for j in range(highY2):
                if ima4b[i, j] == 1:
                    sa_one += 1
                if ima4b[i, j] == 2:
                    sa_two += 1

        self.sym_less_pc = float(sa_one) / sa_two

    def _round(self, x):
        f = np.vectorize(int)
        return f(np.round(x))

    def content_read(self, f1, f2, n):
        sm = 0.0
        x_max = float('-Inf')
        x_min = float('Inf')
        y_max = float('-Inf')
        y_min = float('Inf')

        for i in range(n):
            if f1[i] > x_max:
                x_max = f1[i]
            if f1[i] < x_min:
                x_min = f1[i]
            if f2[i] > y_max:
                y_max = f2[i]
            if f2[i] < y_min:
                y_min = f2[i]

            self.ima[int(f1[i]), int(f2[i])] = 1
            self.img[int(f1[i]), int(f2[i])] = RED


        cx = int(np.average(f1))
        cy = int(np.average(f2))

        # print(len(f1))
        # print(len(f2))
        # print('cx:', cx)
        # print('cy:', cy)
        # print('average:', np.average(self.img_gray))

        self.ima[cx][cy] = int(self.img_gray[cx, cy])
        # print('centro', int(self.img_gray[cx, cy]))
        sm += self.ima[cx][cy] * np.log(self.ima[cx][cy])

        self.vx.append(cx)
        self.vy.append(cy)

        self.wEnt = np.zeros(256, dtype=np.float)
        sw2 = 0

        # print('x: ', x_min, x_max, "y:", y_min, y_max)
        #
        # print('size vx:', len(self.vx))

        k = 0
        while k < len(self.vx):
            lx = self.vx[k]
            ly = self.vy[k]
            if lx > int(x_min)-1 and lx < int(x_max)+1 and ly>int(y_min)-1 and ly < int(y_max)+1:
                self.contour_and_entropy(lx + 1, ly)
                self.contour_and_entropy(lx - 1, ly)
                self.contour_and_entropy(lx, ly + 1)
                self.contour_and_entropy(lx, ly - 1)
            else:
                sw2 = 1
            k += 1

        if sw2 == 0:
            sm = 0.0
            for i in range(256):
                self.wEnt[i] = self.wEnt[i] / float(len(self.vx))
                if self.wEnt[i] > 0:
                    sm = sm + self.wEnt[i] * np.log(self.wEnt[i])
            self.obj_entropy = sm*sm/2.0
            self.obj_size = len(self.vx)
        else:
            self.obj_entropy = 0.0
            self.obj_size = 0.0

        # print('entropy:', self.obj_entropy)
        # print('size:', self.obj_size)
        #
        # print('height', self.height, 'width', self.width)
        #

        # print('pixel', self.img[65, 135]) #  [240 254 243]
        # print('gray: ', self.img_gray[65, 135]) # 249 here, 250 c++

        # print('pixel', self.img[65, 136])
        # print('gray: ', self.img_gray[65, 136])
        #
        #
        # for i in range(self.height):
        #     print('aaa')
        #     for j in range(self.width):
        #         print(i, j, self.ima[i][j], end=', ')
        #
        # print(self.ima.shape)

    def contour_and_entropy(self, i, j):
        if self.ima[i, j] == 0:
            self.vx.append(i)
            self.vy.append(j)
            self.ima[i, j] = self.img_gray[i, j]
            self.wEnt[self.ima[i, j]] = self.wEnt[self.ima[i, j]] + 1


def generate_similarity_classifier_fowl():
    kl = []
    for i in range(1, 8):
        filename = 'kl9596_%d.txt' % (i)
        kl.append(read_csv(basedir, filename))
    ml_w = read_csv(basedir, 'ml9596.txt')
    acerto_medio = [25.637479,
                    26.916101,
                    25.665415,
                    27.480373,
                    25.245048,
                    25.213264,
                    25.585858]
    pw = np.repeat(0.14285, 7)
    return ClassificaGauss(kl, ml_w, acerto_medio, pw, fowl_species)

def generate_similarity_classifier_rabbit():
    kl = []
    for i in range(1, 12):
        filename = 'klrabbit_%d.txt' % (i)
        kl.append(read_csv(basedir, filename))
    ml_w = read_csv(basedir, 'mlrabbit.txt')
    acerto_medio = [19.302075,
                    27.880435,
                    22.425938,
                    21.380911,
                    23.390403,
                    22.006214,
                    17.269468,
                    20.519256,
                    22.786217,
                    19.94028,
                    21.71183]
    pw = np.repeat(0.090909091, 11)
    return ClassificaGauss(kl, ml_w, acerto_medio, pw, rabbit_species)


class ClassificaGauss(object):
    def __init__(self, kl, ml_w, acerto_medio, pw, species):
        self.kl = kl
        self.ml_w = ml_w
        self.acerto_medio = acerto_medio
        self.pw = pw
        self.species = species

    def classify(self, x):
        class_density_value = []
        for i, kl_w in enumerate(self.kl):
            class_density_value.append(self._find_class_density(x, kl_w, i + 1))

        # for x in class_density_value:
        #     print('density:', x)

        taxa_acerto = np.zeros(7, dtype=np.float)
        for i in range(7):
            if class_density_value[i] > 0.0:
               taxa_acerto[i] = class_density_value[i] * 100. / self.acerto_medio[i]

        classification = dict()
        for i in reversed(np.argsort(taxa_acerto)):
            if taxa_acerto[i] > 0.0:
                classification[self.species[i]] = taxa_acerto[i]

        return classification

    def _find_class_density(self, x, kl_w, w_especie):
        gx = .0

        if not math.isclose(np.linalg.det(kl_w), 0): # det(kl_w) != 0.0
            mx = np.zeros((1, 13), dtype=np.float)
            mxt = np.zeros((13, 1), dtype=np.float)
            for i in range(13):
                mx[0, i] = x[i] - self.ml_w[w_especie-1, i]
                # print('x_i:', x[i])
                # print('subtraendo:', self.ml_w[w_especie-1, i])
                # print('mx:', mx[0, i])
                mxt[i, 0] = x[i] - self.ml_w[w_especie-1, i]
            mx_inv = np.dot(mx, np.linalg.inv(kl_w))
            mx_inv_mx = np.dot(mx_inv, mxt)

            # print('mx shape', mx.shape)
            # print('inv shape', np.linalg.inv(kl_w).shape)
            # print('mx_inv', mx_inv.shape)
            #
            # print('x', x)
            # print('mx', mx)

            aa = mx_inv_mx[0, 0]
            # print('aa:', aa)
            bb = np.linalg.det(kl_w)
            # print('det:', bb)
            cc = np.log(bb)
            # cc = round(cc, 4)
            # print('log:', cc)

            # print ('aa:', aa, ' bb:', bb, ' cc:', cc)
            gx = (-0.5) * aa - (0.5 * cc)
            if not math.isclose(self.pw[w_especie-1], 0.0):
                gx = gx + np.log(self.pw[w_especie-1])

        # print('gx: ', gx)
        return gx


def generate_probability_classifier_rabbit():
    fq = []
    for i in range(1, 14):
        filename = 'freqRabbit_%d.txt' % (i)
        fq.append(np.array(read_csv(basedir, filename), dtype=np.float64))
    per_w = read_csv(basedir, 'PerRabbit.txt')
    vpriori = np.repeat(0.090909091, 11)
    return ClassificaProb(fq, per_w, vpriori, rabbit_species)


def generate_probability_classifier_fowl():
    fq = []
    for i in range(1, 14):
        filename = 'freqFowl_%d.txt' % (i)
        fq.append(np.array(read_csv(basedir, filename), dtype=np.float64))
    per_w = read_csv(basedir, 'PerFowl.txt')
    vpriori = np.repeat(0.14285, 7)
    return ClassificaProb(fq, per_w, vpriori, fowl_species)


class ClassificaProb:
    def __init__(self, fq, per_w, vpriori, species):
        self.fq = fq
        self.per_w = per_w
        self.vpriori = vpriori
        self.species = species
        self.nclass = len(species)

    def classify(self, x):
        self._find_posteriori(x, self.fq[0], self.fq[0], 0)
        for i in range(1, 13):
            self._find_posteriori(x, self.fq[i - 1], self.fq[i], i)

        """
        The last frequency matrix stores the final classification results; 
        detection is done locating the percetil where the last feature is.
        Then, the column of the percentil elected is the posterior probability
        classification.
        """
        wflag = False
        taxa_acerto = np.zeros(self.nclass, dtype=np.float)
        for wcont in range(self.nclass):
            wper = self.per_w[12, wcont]
            if not wflag and x[12] <= wper:
                for i in range(self.nclass):
                    taxa_acerto[i] = self.fq[-1][i, wcont] * 100
                wflag = True

        if not wflag:
            """
            If the element is greater than higher value, it is considered
            in last percentil
            """
            for i in range(self.nclass):
                taxa_acerto[i] = self.fq[-1][i, -1] * 100

        classification = dict()
        for i in reversed(np.argsort(taxa_acerto)):
            if taxa_acerto[i] > 1e-4:
                classification[self.species[i]] = taxa_acerto[i]

        return classification

    def _find_posteriori(self, x, fq0, fq2, w_feature):
        """
        Computes the posterior probability of the frequency matrix; this approach
        is based on the Dirichlet density (frequency and percentiles matrices).
        :param x: features vector
        :param fq0: previous frequency matrix
        :param fq2: current frequency matrix
        :param w_feature:
        """
        wsum = 0.0
        aa = 0.0
        wper = 0.0


        # TODO: acho que é possível simplificar os for's
        for i in range(self.nclass):
            wsum = 0.0
            for j in range(self.nclass):
                aa = fq2[i, j]
                aa = aa * (2.0 / self.nclass)
                fq2[i, j] = aa
                wsum += aa
            for j in range(self.nclass):
                aa = fq2[i, j]
                if wsum > 0.0:
                    aa = aa / wsum
                fq2[i, j] = aa

        if w_feature == 0:
            for i in range(self.nclass):
                wsum = 0.0
                for j in range(self.nclass):
                    aa = fq2[j, i]
                    aa = aa * self.vpriori[j]
                    fq2[j, i] = aa
                    wsum += aa
                for j in range(self.nclass):
                    aa = fq2[j, i]
                    if wsum > 0.0:
                        aa = aa / wsum
                    fq2[j, i] = aa
        else:
            wflag = False
            for wcont in range(self.nclass):
                """
                if the number of features is greater than 0,
                the correct percentil was found in the previous matrix
                and the column-percentil will be the priori probability
                """
                wper = self.per_w[w_feature-1, wcont]
                if not wflag and x[w_feature-1] <= wper:
                    for i in range(self.nclass):
                        self.vpriori[i] = fq0[i, wcont]
                    wflag = True
            if not wflag:
                """
                if the element is greater than the highest value, it is 
                connsidered in last percentil
                """
                for i in range(self.nclass):
                    self.vpriori[i] = fq0[i, self.nclass-1]
            for i in range(self.nclass):
                wsum = 0.0
                for j in range(self.nclass):
                    """
                    frequency matrix is multiplied by the new priori 
                    probability vector, computed from the previous matrix
                    """
                    aa = fq2[j, i]
                    aa = aa * self.vpriori[j]
                    fq2[j, i] = aa
                    wsum += aa
                for j in range(self.nclass):
                    aa = fq2[j, i]
                    if wsum > 0.0:
                        aa = aa / wsum
                    fq2[j, i] = aa


def read_csv(basedir, filename):
    return np.array(pd.read_csv('%s/%s'%(basedir, filename), sep='\s+', header=None).as_matrix(), dtype=np.float64)
