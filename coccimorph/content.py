import numpy as np
from coccimorph.segment import load_image


RED = (0, 0, 255)

class FeatureExtractor:
    def __init__(self, filename, scale):
        self.img = load_image(filename, scale)
        self.height, self.width, _ = self.img.shape
        self.img_gray = np.array(np.average(self.img, axis=2), dtype=np.int)
        self.ima = np.zeros((self.height, self.width), dtype=np.int)
        self.vx = []
        self.vy = []
        self.wEnt = None

        self.obj_entropy = 0.0
        self.obj_size = 0.0

    def set_co_matrix(self, d: int):
        aux_mcc = np.zeros((256, 256), dtype=np.int)
        ro = 0
        for x in range(self.height):
            for y in range(self.width-d):
                if self.ima[x,y] > 0 and self.ima[x,y+d] > 0:
                    aux_mcc[self.ima[x,y],self.ima[x,y+d]] += 1
                    ro += 1

        y = 0
        for x in range(self.height):
            y = self.width-1
            while y > d-1:
                if self.ima[x,y]>0 and self.ima[x,y-d]>0:
                    aux_mcc[self.ima[x,y], self.ima[x,y-d]] += 1
                    ro += 1
                y -= 1

        self.mcc = aux_mcc / float(ro)

    def mcc_asm(self):
        return np.sum(np.power(self.mcc, 2))

    def mcc_con(self):
        sm = 0.0
        for i in range(256):
            for j in range(256):
                sm += self.mcc[i,j]*(i-j)*(i-j)
        return sm

    def mcc_idf(self):
        sm = 0.0
        for i in range(256):
            for j in range(256):
                sm += self.mcc[i,j] / float(1 + (i-j)*(i-j))
        return sm

    def mcc_ent(self):
        sm = 0.0
        for i in range(256):
            for j in range(256):
                if self.mcc[i,j]>0:
                    sm += self.mcc[i,j]*np.log(self.mcc[i,j])
        return sm * sm / 2.

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

        # compute eigen vectors and eigen values
        eigenvalues, eigenvectors = np.linalg.eigh(k)

        evec_inv = np.linalg.inv(eigenvectors)

        # transform to new space using inverse matrix of eigen vectors
        vx1 = np.zeros(n, dtype=np.float)
        vy1 = np.zeros(n, dtype=np.float)
        vx2 = np.zeros(n, dtype=np.float)
        vy2 = np.zeros(n, dtype=np.float)
        sumvx1 = 0
        sumvy1 = 0
        for i in range(n):
            vx_w = evec_inv[0,0]*self.vx[i] + evec_inv[0,1]*self.vy[i]
            vy_w = evec_inv[1,0]*self.vx[i] + evec_inv[1,1]*self.vy[i]
            sumvx1 += vx_w
            sumvy1 += vy_w
            vx1[i] = vx_w
            vy1[i] = vy_w

        meanvx1 = sumvx1 / float(n)
        meanvy1 = sumvy1 / float(n)

        vx1 = vx1 - meanvx1
        vy1 = vy1 - meanvy1
        vx2 = np.copy(vx1)
        vy2 = np.copy(vy1)

        # searching for diameters
        highX = np.max(vx1)
        lessX = np.min(vx1)
        highY = np.max(vy1)
        lessY = np.min(vy1)

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

        # compute the simmetry
        highX1 = np.max(self._round(vx1))+3
        highY1 = np.max(self._round(vy1))+3
        highX2 = np.max(self._round(vx2))+3
        highY2 = np.max(self._round(vy2))+3

        # create temporal matrices to compute erosion, dilation and rate simmetry
        ima3a = np.zeros((highX1, highY1))
        ima3b = np.zeros((highX2, highY2))

        ima2a = np.zeros((highX1, 500), dtype=np.int)
        ima2b = np.zeros((highX2, 500), dtype=np.int)
        ima4a = np.zeros((highX1, 500), dtype=np.int)
        ima4b = np.zeros((highX2, 500), dtype=np.int)

        for i in range(n):
            ima2a[int(self.vx[i]), int(self.vy[i])] = 1
            ima2b[int(self.vx[i]), int(self.vy[i])] = 1
            # print(ima3a.shape)
            # print(int(np.round(vy1[i])))
            ima3a[int(np.round(vx1[i])), int(np.round(vy1[i]))] = 1
            ima3b[int(np.round(vx2[i])), int(np.round(vy2[i]))] = 1

        ima3a = self.dilate(ima3a)
        ima3a = self.erode(ima3a)
        for i in range(highX1):
            for j in range(highY1):
                ima4a[i, j] = ima2a[i, j] + ima3a[i, j]

        ima3b = self.dilate(ima3b)
        ima3b = self.erode(ima3b)
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

    def erode(self, ima: np.ndarray):
        dx, dy = ima.shape
        ima_temp = np.zeros((dx, 500), dtype=np.int)

        for m in range(dx):
            for n in range(dy):
                ima_temp[m, n] = ima[m, n]

        for m in range(1, dx-1):
            for n in range(1, dy-1):
                if ima_temp[m, n] == 1:
                    aux = 1
                    aux *= ima_temp[m, n]
                    aux *= ima_temp[m-1, n]
                    aux *= ima_temp[m+1, n]
                    aux *= ima_temp[m, n-1]
                    aux *= ima_temp[m, n+1]
                    ima[m, n] = aux

        for i in range(dx):
            ima[i, 0] = 0
            ima[i, dy-1] = 0

        for i in range(dy):
            ima[0, i] = 0
            ima[dx-1, i] = 0

        return ima

    def dilate(self, ima):
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

        for m in range(1, dx-1):
            for n in range(1, dy-1):
                if ima_temp[m, n] == 1:
                    for i in range(3):
                        for j in range(3):
                            mw = m-1
                            nw = n-1
                            if ima[mw+i, nw+j] == 0:
                                ima[mw+i, nw+j] = ima[mw+i, nw+j] or se[i][j]
        return ima

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

        print(len(f1))
        print(len(f2))
        print('cx:', cx)
        print('cy:', cy)

        self.ima[cx][cy] = int(np.average(self.img[cx, cy]))
        sm += self.ima[cx][cy] * np.log(self.ima[cx][cy])

        self.vx.append(cx)
        self.vy.append(cy)

        self.wEnt = np.zeros(256, dtype=np.float)
        sw2 = 0

        print('x: ', x_min, x_max, "y:", y_min, y_max)

        print('size vx:', len(self.vx))

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

        print('entropy:', self.obj_entropy)
        print('size:', self.obj_size)

    def contour_and_entropy(self, i, j):
        if self.ima[i, j] == 0:
            self.vx.append(i)
            self.vy.append(j)
            self.ima[i, j] = self.img_gray[i, j]
            self.wEnt[self.ima[i, j]] = 1 + self.wEnt[self.ima[i, j]]

