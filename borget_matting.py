import numpy as np
import cv2

COE = 10000
stripwidth = 6
L = 20
MAXNUM = 9999999
sigmaLevels = 15
deltaLevels = 11

nstep = 8
nx = [0, 1, 0, -1, -1, -1, 1, 1]
ny = [1, 0, -1, 0, -1, 1, -1, 1]

rstep = 4
rx = [0, 1, 0, -1]
ry = [1, 0, -1, 0]


class point:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class para_point:
    delta = None
    sigma = None

    def __init__(self, point, index, sec):
        self.p = point
        self.idx = index
        self.sec = sec


class inf_point:
    def __init__(self, point, dis, area):
        self.p = point
        self.dis = dis
        self.area = area


class dands:
    def __init__(self, delta, sigma):
        self.delta = delta
        self.sigma = sigma


def show(img):
    cv2.imshow("title", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def outrange(x, l, r):
    if (x < l or x > r):
        return True
    else:
        return False


def border_detection(img):
    edge = cv2.Canny(img, 100, 200)
    return edge


def Vfunc(ddelta, dsigma):
    global lamda1, lamda2
    return (lamda1 * pow(ddelta, 2.0) + lamda2 * pow(dsigma, 2.0)) / 200


def dfs(x, y):
    global edge, color, areacnt, sections, contour, nstep, nx, ny, rows, cols
    color[x, y] = 255
    areacnt += 1
    p = point(x, y)
    pt = para_point(p, areacnt, sections)
    contour.append(pt)
    for i in range(nstep):
        zx = nx[i]
        zy = ny[i]
        newx = x + zx
        newy = y + zy
        if outrange(newx, 0, rows - 1) or outrange(newy, 0, cols - 1):
            continue
        if edge[newx, newy] == 0 or color[newx, newy] != 0:
            continue
        dfs(newx, newy)


def parameterization_contour():
    global rows, cols, edge, color, sections
    for i in range(rows):
        for j in range(cols):
            if edge[i, j] != 0:
                if color[i, j] != 0:
                    continue
                dfs(i, j)
                sections += 1


def strip_init(mask):
    global contour, strip, COE, rstep, rx, ry
    color = np.zeros(mask.shape, dtype=np.int32)
    queue = []

    for i in range(len(contour)):
        ip = inf_point(contour[i].p, 0, contour[i].idx)
        strip["{}".format(ip.p.x * COE + ip.p.y)] = ip
        queue.append(ip.p)
        color[ip.p.x, ip.p.y] = ip.area + 1

    lc = 0
    while (lc < len(queue)):
        p = queue[lc]
        lc += 1
        ip = strip["{}".format(p.x * COE + p.y)]
        if abs(ip.dis) >= stripwidth:
            break
        x = ip.p.x
        y = ip.p.y
        for i in range(rstep):
            newx = x + rx[i]
            newy = y + ry[i]
            if outrange(newx, 0, rows - 1) or outrange(newy, 0, cols - 1):
                continue

            if color[newx, newy] != 0:
                continue

            nip_point = point(newx, newy)
            nip = inf_point(nip_point, abs(ip.dis) + 1, ip.area)

            if (mask[newx, newy] & 1) != 1:
                nip.dis = -nip.dis

            strip["{}".format(nip.p.x * COE + nip.p.y)] = nip
            queue.append(nip.p)
            color[newx, newy] = nip.area + 1


def gaussian(x, delta, sigma):
    e = np.exp(-(pow(x - delta, 2.0) / (2.0 * sigma)))
    rs = 1.0 / (pow(sigma, 0.5) * pow(2.0 * np.pi, 0.5)) * e
    return rs


def ufunc(a, uf, ub):
    return (1.0 - a) * ub + a * uf


def cfunc(a, cf, cb):
    return pow(1.0 - a, 2.0) * cb + pow(a, 2.0) * cf


def sigmoid(r, delta, sigma):
    rs = -(r - delta) / sigma + 1e-7
    rs = np.exp(rs)
    rs = 1.0 / (1.0 + rs)
    return rs


def d_term(ip, I, delta, sigma):
    global stripwidth, uf, ub, cf, cb

    alpha = sigmoid(ip.dis / stripwidth, delta, sigma)
    D = gaussian(I, ufunc(alpha, uf, ub), cfunc(alpha, cf, cb))
    D = -np.log(D) / np.log(2.0)
    return D


def d_func(index, p, delta, sigma, gray):
    global strip, stripwidth, rx, ry, rstep, uf, ub, cf, cb

    queue = []
    color = {}
    total = 0
    ip = strip["{}".format(p.x * COE + p.y)]
    total += d_term(ip, gray[ip.p.x, ip.p.y], delta, sigma)
    queue.append(ip)
    color["{}".format(ip.p.x * COE + ip.p.y)] = True

    lc = 0
    while (lc < len(queue)):
        ip = queue[lc]
        lc += 1
        if abs(ip.dis) >= stripwidth:
            break
        x = ip.p.x
        y = ip.p.y
        for i in range(rstep):
            newx = x + rx[i]
            newy = y + ry[i]
            if outrange(newx, 0, rows - 1) or outrange(newy, 0, cols - 1):
                continue
            try:
                if color["{}".format(newx * COE + newy)]:
                    continue
            except:
                pass
            newip = strip["{}".format(newx * COE + newy)]
            if (newip.area == index):
                total += d_term(newip, gray[newx, newy], delta, sigma)

            queue.append(newip)
            color["{}".format(newx * COE + newy)] = True
    return total


def cal_sample_mean_covariance(p, gray, mask):
    global L, rows, cols, uf, ub, cf, cb

    sumf = 0
    sumb = 0
    cntf = 0
    cntb = 0

    for x in range(p.x - L, p.x + L):
        for y in range(p.y - L, p.y + L):
            if not(outrange(x, 0, rows - 1) or outrange(y, 0, cols - 1)):
                    g = gray[x, y]
                    if mask[x, y] == 0:
                        sumb += g
                        cntb += 1

                    else:
                        sumf += g
                        cntf += 1

    uf = sumf / cntf
    ub = sumb / cntb

    cf = 0
    cb = 0
    for x in range(p.x - L, p.x + L):
        for y in range(p.y - L, p.y + L):
            if not(outrange(x, 0, rows - 1) or outrange(y, 0, cols - 1)):
                g = gray[x, y]
                if mask[x, y] == 0:
                    cb += pow(g - ub, 2.0)

                else:
                    cf += pow(g - uf, 2.0)

    cf /= cntf
    cb /= cntb


def sigma(level):
    return (0.025 * level) + 1e-7


def delta(level):
    return 0.1 * level


def energy_minimization(org_img, mask):
    global ef, uf, ub, cf, cb, areacnt, vecds

    gray = cv2.cvtColor(org_img, cv2.COLOR_RGB2GRAY) / 255.
    for i in range(len(contour)):
        pp = contour[i]
        index = pp.idx

        cal_sample_mean_covariance(pp.p, gray, mask)
        for d0 in range(deltaLevels):
            for s0 in range(sigmaLevels):
                sigma0 = sigma(s0)
                delta0 = delta(d0)
                ef[index][d0][s0] = MAXNUM

                D = d_func(index, pp.p, delta0, sigma0, gray)
                if index == 0:
                    ef[index][d0][s0] = D
                    continue

                for d1 in range(deltaLevels):
                    for s1 in range(sigmaLevels):
                        delta1 = delta(d1)
                        sigma1 = sigma(s1)
                        Vterm = 0
                        if contour[i - 1].sec == pp.sec:
                            Vterm = Vfunc(delta0 - delta1, sigma0 - sigma1)

                        rs = ef[index - 1][d1][s1] + Vterm + D
                        if rs < ef[index][d0][s0]:
                            ds = dands(d1, s1)
                            ef[index][d0][s0] = rs
                            rec[index][d0][s0] = ds

    minE = MAXNUM
    for d0 in range(deltaLevels):
        for s0 in range(sigmaLevels):
            if ef[areacnt - 1][d0][s0] < minE:
                minE = ef[areacnt - 1][d0][s0]
                ds = dands(d0, s0)

    vecds["{}".format(areacnt - 1)] = ds
    for i in range(areacnt - 2, -1, -1):
        ds0 = vecds["{}".format(i + 1)]
        ds = rec[i + 1][ds0.delta][ds0.sigma]
        vecds["{}".format(i)] = ds


def adjust_a(a):
    if a < 0.01:
        return 0
    if a > 9.99:
        return 1
    return areacnt


def CalculateMask(mask):
    global contour, vecds, stripwidth, rstep
    bordermask = np.zeros_like(mask, dtype=np.float32)
    color = np.zeros_like(mask, dtype=np.int32)

    queue = []
    for i in range(len(contour)):
        ip = inf_point(contour[i].p, 0, contour[i].idx)
        queue.append(ip)
        color[ip.p.x, ip.p.y] = 1

        ds = vecds["{}".format(ip.area)]
        alpha = sigmoid(ip.dis / stripwidth, delta(ds.delta), sigma(ds.sigma))
        alpha = adjust_a(alpha)
        bordermask[ip.p.x, ip.p.y] = float(alpha)

    lc = 0
    while (lc < len(queue)):
        ip = queue[lc]
        lc += 1
        x = ip.p.x
        y = ip.p.y
        for i in range(rstep):
            newx = x + rx[i]
            newy = y + ry[i]
            if outrange(newx, 0, rows - 1) or outrange(newy, 0, cols - 1):
                continue
            if color[newx, newy] != 0:
                continue
            np = point(newx, newy)
            nip = inf_point(np, abs(ip.dis) + 1, ip.area)
            if mask[newx, newy] != 1:
                nip.dis = -nip.dis

            queue.append(nip)
            color[newx, newy] = 1

            ds = vecds["{}".format(nip.area)]
            alpha = sigmoid(nip.dis / stripwidth, delta(ds.delta), sigma(ds.sigma))
            alpha = adjust_a(alpha)
            bordermask[nip.p.x, nip.p.y] = float(alpha)


org_img = cv2.imread("images/dog.jpg")
mask = cv2.imread("images/mask.jpg", 0)

lamda1 = 50
lamda2 = 1000

ef = np.zeros((5000, deltaLevels, sigmaLevels))
rec = np.zeros((5000, deltaLevels, sigmaLevels), dtype=dands)

rows = org_img.shape[0]
cols = org_img.shape[1]

sections = 0
areacnt = 0
tot = 0
contour = []
strip = {}
vecds = {}

img = cv2.imread("temp.jpg")
edge = mask
edge = border_detection(edge)
color = np.zeros(edge.shape, dtype=np.uint8)

parameterization_contour()

tmask = mask
strip_init(tmask)

energy_minimization(org_img, mask)

border_mask = CalculateMask(mask)
border_mask = cv2.GaussianBlur(border_mask, (7, 7), 9)
