import numpy as np
import cv2
from tqdm.auto import tqdm


def in_range(x, _max):
    return min(max(x, 0), _max)


class point2d:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y


class point:
    def __init__(self, _x, _y, _dis=0):
        self.p = point2d(_x, _y)
        self.dis = int(_dis)
        self.delta = 0
        self.sigma = 0
        self.alpha = 1e-9
        self.para = LocalPara()

    def distance(self, t):
        return np.sqrt(pow((self.p.x - t.p.x), 2) + pow((self.p.y - t.p.y), 2))


class Contour:
    def __init__(self, _p):
        self.p = _p
        self.neighbor = []


class LocalPara:
    def __init__(self):
        self.Bmean = np.zeros([3], dtype=np.uint8)
        self.Fmean = np.zeros([3], dtype=np.uint8)
        self.Bvar = 1e-9
        self.Fvar = 1e-9


class mybm:
    def __init__(self, _originImage, _mask):
        self.Src = _originImage
        self.rows = self.Src.shape[0]
        self.cols = self.Src.shape[1]
        self.Mask = _mask
        self.Mask = np.array((self.Mask >= 245), dtype=np.uint8)
        self.Edge = cv2.Canny(self.Mask, 0, 1)
        self.g = np.zeros([6, 30, 10], dtype=np.float32)
        self.contour = []
        self.Oldcontour = []
        for i in range(6):
            for j in range(30):
                for k in range(10):
                    self.g[i][j][k] = self.linear(i, j * 0.2, k * 0.6)
        print("Construct Contour Start!")
        self.ConstructContour()
        print("Construct Contour Done!\nConstruct Strip Start!")
        self.ConstructStrip()
        print("Construct Strip Done!")

    def linear(self, x, _delta, _sigma):
        k = 0.0
        if _sigma >= 0.1:
            k = 1 / _sigma
        if x < _delta - (_sigma / 2):
            return 0
        if x >= _delta + _sigma / 2:
            return 1
        return 0.5 + k * (x - _delta)

    def ConstructStrip(self):
        _edge = self.Edge
        _mask = self.Mask
        for i in tqdm(range(self.rows), ascii=True):
            for j in range(self.cols):
                if _edge[i, j] == 0 and _mask[i, j] == 1:
                    p = point(i, j)
                    dis = self.CheckDis(p)
                    if dis > 0:
                        self.push2(p, 3)

    def ConstructContour(self):
        for i in tqdm(range(self.rows), ascii=True):
            for j in range(self.cols):
                if self.Edge[i, j]:
                    self.push1(point(i, j), self.contour, 2)

    def CheckDis(self, p):
        for _contour in self.contour:
            if p.distance(_contour.p) < 3:
                return p.distance(_contour.p)
        return -1

    def push1(self, p, _list, threshold):
        for i in range(len(_list)):
            if (p.distance(_list[i].p) <= threshold):
                _list.insert(i, Contour(p))
                return
        _list.append(Contour(p))

    def push2(self, p, threshold):
        _min = -1
        for i in range(len(self.contour)):
            if (p.distance(self.contour[i].p) < threshold):
                _min = i
                threshold = p.distance(self.contour[i].p)
        if (_min == -1):
            return
        else:
            p.dis = int(threshold)
            self.contour[_min].neighbor.append(p)

    def getLocalMandV(self, p, rst):
        x = p.p.x - 20
        if x < 0:
            x = 0
        xlen = x + 41
        if xlen < self.cols:
            xlen = 41
        else:
            xlen = self.cols - x

        y = p.p.y - 20
        if y < 0:
            y = 0
        ylen = y + 41
        if ylen < self.rows:
            ylen = 41
        else:
            ylen = self.rows - y
        neibor = self.Src[x - xlen // 2:x + xlen // 2, y - ylen // 2:y + ylen // 2]
        Bmean = np.zeros([3], dtype=np.uint8)
        Fmean = np.zeros([3], dtype=np.uint8)
        Bvar = 0.0
        Fvar = 0.0
        Fn = 0
        Bn = 0
        for i in range(neibor.shape[0]):
            for j in range(neibor.shape[1]):
                tmp = neibor[i, j]
                if self.Edge[in_range(y + i, self.rows - 1), in_range(x + j, self.cols - 1)] == 255:
                    Fmean += tmp
                    Fn += 1
                else:
                    Bmean += tmp
                    Bn += 1
        if Fn:
            Fmean = Fmean / Fn
        else:
            Fmean = np.zeros([3], dtype=np.uint8)
        if Bn:
            Bmean = Bmean / Bn
        else:
            Bmean = np.zeros([3], dtype=np.uint8)
        for i in range(neibor.shape[0]):
            for j in range(neibor.shape[1]):
                tmp = neibor[i, j]
                if self.Edge[in_range(y + i, self.rows - 1), in_range(x + j, self.cols - 1)] == 255:
                    Fvar += np.dot((Fmean - tmp), (Fmean - tmp))
                else:
                    Bvar += np.dot((tmp - Bmean), (tmp - Bmean))

        if (Fn):
            Fvar = Fvar / Fn
        else:
            Fvar = 0
        if (Bn):
            Bvar = Bvar / Bn
        else:
            Bvar = 0
        rst.Bmean = Bmean
        rst.Bvar = Bvar
        rst.Fmean = Fmean
        rst.Fvar = Fvar

    def Gaussian(self, _x, _delta, _sigma):
        e = np.exp(-(pow(_x - _delta, 2.0) / (2.0 * (_sigma + 1e-9))))
        rs = 1.0 / np.sqrt(_sigma + 1e-9) * np.sqrt(2.0 * np.pi) * e
        return rs

    def Mmean(self, x, Fmean, Bmean):
        return (1.0 - x) * Bmean + x * Fmean

    def Mvar(self, x, Fvar, Bvar):
        return (1.0 - x) * (1.0 - x) * Bvar + x * x * Fvar

    def toGray(self, tmp):
        return int((tmp[2] * 299 + tmp[1] * 587 + tmp[0] * 114 + 500) / 1000)

    def dataTermPoint(self, _ip, _I, _delta, _sigma, para):
        alpha = self.g[_ip.dis][_delta][_sigma]
        D = self.Gaussian(_I, self.Mmean(alpha, self.toGray(para.Fmean), self.toGray(para.Bmean)), self.Mvar(alpha, para.Fvar, para.Bvar))
        D = -np.log(D + 1e-9) / np.log(2.0)
        return D

    def Run(self):
        Emin = -1
        delta = 15
        sigma = 5
        for di in tqdm(range(30), ascii=True):
            for si in tqdm(range(10), ascii=True):
                D = self.dataTermPoint(self.contour[0].p, self.toGray(self.Src[self.contour[0].p.p.y, self.contour[0].p.p.x]), di, si, self.contour[0].p.para)
                for j in range(len(self.contour[0].neighbor)):
                    p = self.contour[0].neighbor[j]
                    self.getLocalMandV(p, p.para)
                    D += self.dataTermPoint(p, self.toGray(self.Src[p.p.y, p.p.x]), di, si, p.para)
                if D < Emin:
                    Emin = D
                    delta = di
                    sigma = si
                for i in range(len(self.contour)):
                    para = LocalPara()
                    self.getLocalMandV(self.contour[i].p, para)
                    self.contour[i].p.para = para
                    for j in range(len(self.contour[i].neighbor)):
                        p = self.contour[i].neighbor[j]
                        self.getLocalMandV(p, para)
                        p.para = para
                    _min = 1e+9
                    for si in range(30):
                        for di in range(10):
                            D = self.dataTermPoint(self.contour[i].p, self.toGray(self.Src[self.contour[i].p.p.y, self.contour[i].p.p.x]), si, di, self.contour[i].p.para)
                            for j in range(len(self.contour[i].neighbor)):
                                p = self.contour[i].neighbor[j]
                                D += self.dataTermPoint(p, self.toGray(self.Src[p.p.y, p.p.x]), si, di, p.para)
                            V = 2 * (si - delta) * (si - delta) + 360 * (sigma - di) * (sigma - di)
                            if (D + V < _min):
                                _min = D + V
                                self.contour[i].p.delta = si
                                self.contour[i].p.sigma = di
                    sigma = self.contour[i].p.sigma
                    delta = self.contour[i].p.delta
                    self.contour[i].p.alpha = self.g[0][delta][sigma]
                    for j in range(len(self.contour[i].neighbor)):
                        p = self.contour[i].neighbor[j]
                        p.alpha = self.g[p.dis][delta][sigma]
        _alphaMask = np.zeros_like(self.Mask, dtype=np.float32)
        for i in range(self.Mask.shape[0]):
            for j in range(self.Mask.shape[1]):
                _alphaMask[i, j] = self.Mask[i, j]
        for i in range(len(self.contour)):
            _alphaMask[self.contour[i].p.p.y, self.contour[i].p.p.x] = self.contour[i].p.alpha
            for j in range(len(self.contour.neighbor)):
                p = self.contour[i].neighbor[j]
                _alphaMask[p.p.y, p.p.x] = p.alpha
        rst = np.zeros_like(self.Src, dtype=np.uint8)
        for i in range(rst.shape[0]):
            for j in range(rst.shape[1]):
                rst[i, j] = self.Src[i, j] * _alphaMask[i, j] * 255

        cv2.imwrite("done.png", rst)
        print("Done!")


if __name__ == '__main__':
    org_img = cv2.imread("images/llama_resize.jpg")
    mask = cv2.imread("images/llama_resize_mask.jpg", 0)
    inst = mybm(org_img, mask)
    print("Run Start!")
    inst.Run()
    print("Run Done!")
