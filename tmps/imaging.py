#! /user/bin/env python3
# -*- coding: utf-8 -*-

# Module for tiff images

# By Logan Hillberry

import json
import copy
import numpy as np
import matplotlib.pyplot as plt  # requires pillow
import matplotlib.patches as patches
from scipy.ndimage import rotate
import scipy.optimize as opt
import scipy.odr
import os

class Image:
    def __init__(self, path, scale=300):  # scale:px/cm
        self.path = path
        self.scale = scale
        # get grayscal from rgb-style data (assum grayscale camera)
        try:
            im = np.array(
                plt.imread(path)[:, :, 0], dtype=np.float64
            )
        except IndexError:
            print("is gray scale")
            im = plt.imread(path)
        self.im = np.flipud(im)
        self.extent = [0.0, self.shape[1] / self.scale, 0.0, self.shape[0] / self.scale]
        self.xaxis = np.arange(self.extent[0], self.extent[1], 1 / self.scale)
        self.yaxis = np.arange(self.extent[2], self.extent[3], 1 / self.scale)
        self.XY = np.meshgrid(self.xaxis, self.yaxis, indexing="xy")

    @property
    def shape(self):
        return self.im.shape

    def show(self, contours=False):
        fig, ax = plt.subplots(1, 1)
        # ax.pcolormesh(self.XY[0], self.XY[1], self.im)
        ax.imshow(self.im, extent=self.extent, origin="lower")
        ax.set_aspect(aspect="equal")
        if contours:
            data_fitted = self.gaussian_2d(self.XY, *self.popt).reshape(*self.shape)
            ax.contour(self.XY[0], self.XY[1], data_fitted, levels=4, colors="k")
            #ax.title("{}: {}".format(self.pname, self.pval))
        plt.show()

    def rotate(self, ang):
        self.im = rotate(self.im, ang, reshape=False)

    def crop(self, l, r, b, t, units="px"):
        if units == "cm":
            l, r, b, t = map(
                int, [l * self.scale, r * self.scale, b * self.scale, t * self.scale]
            )
            x0 = int(self.xaxis[0] * self.scale)
            y0 = int(self.yaxis[0] * self.scale)
            Ny = self.im.shape[0]
            l -= x0
            r -= x0
            t -= y0
            b -= y0
        else:
            x0, y0 = 0, 0
        if units in ("cm", "px"):
            #print(self.shape, l, r, b, t)
            assert 0 <= r <= self.shape[1]
            assert 0 <= t <= self.shape[0]
            assert 0 <= l <= self.shape[1]
            assert 0 <= b <= self.shape[0]
            self.xaxis = self.xaxis[l:r]
            self.yaxis = self.yaxis[b:t]
            self.XY = np.meshgrid(self.xaxis, self.yaxis, indexing="xy")
            self.im = self.im[b:t, l:r]
            self.extent = [
                self.xaxis[0],
                self.xaxis[-1],
                self.yaxis[0],
                self.yaxis[-1],
            ]
        else:
            raise ValueError("unknown units {}".format(units))

    def moments(self):
        image = self.im
        X, Y = self.XY
        # x and y step size
        dx = X[0][1] - X[0][0]
        dy = Y[1][0] - Y[0][0]
        # x and y initial value
        xa = X[0][0]
        ya = Y[0][0]
        # x and y first moment
        x0 = (X * image).sum() / image.sum()
        y0 = (Y * image).sum() / image.sum()
        # x and y indicies of first moment
        yidx = int((y0 - ya) / dy)
        xidx = int((x0 - xa) / dx)
        # vertical line cut through centroid
        col = image[:, xidx]
        yaxis = Y[:, xidx]
        # horizontal line cut through centroid
        row = image[yidx, :]
        xaxis = X[yidx, :]
        # x and y second moment
        sy = np.sqrt(abs((yaxis - y0) ** 2 * col).sum() / col.sum())
        sx = np.sqrt(abs((xaxis - x0) ** 2 * row).sum() / row.sum())
        # angle, amplitude, and offset estimates
        theta = 0.0
        A = image.max()
        B = image.min()
        return x0, y0, sx, sy, theta, A, B

    def fit_crop(self, N=5):
        self.leastsq_fit()
        x0, y0, sx, sy, ang, A, B = self.popt
        Dx = N * sx / 2
        Dy = N * sy / 2
        l = max(x0 - Dx, self.xaxis[0])
        r = min(x0 + Dx, self.xaxis[-1])
        b = max(y0 - Dy, self.yaxis[0])
        t = min(y0 + Dy, self.yaxis[-1])
        self.crop(l, r, b, t, units="cm")

    # shift axes by x0, y0
    def center(self, x0, y0):
        self.extent = [
            self.extent[0] - x0,
            self.extent[1] - x0,
            self.extent[2] - y0,
            self.extent[3] - y0,
        ]
        self.popt[0] -= x0
        self.popt[1] -= y0
        self.xaxis -= x0
        self.yaxis -= y0
        self.XY = np.meshgrid(self.xaxis, self.yaxis, indexing="xy")

    @staticmethod
    def gaussian_2d(xy, x0, y0, sx, sy, theta, A, B):
        x, y = xy
        x0 = float(x0)
        y0 = float(y0)
        a = (np.cos(theta) ** 2) / (2 * sx ** 2) + (np.sin(theta) ** 2) / (2 * sy ** 2)
        b = -(np.sin(2 * theta)) / (4 * sx ** 2) + (np.sin(2 * theta)) / (4 * sy ** 2)
        c = (np.sin(theta) ** 2) / (2 * sx ** 2) + (np.cos(theta) ** 2) / (2 * sy ** 2)
        g = B + A * np.exp(
            -(a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2))
        )
        return g.ravel()

    def leastsq_fit(self, p0=None):
        errfunc = lambda p, xy, z: self.gaussian_2d(xy, *p) - z
        if p0 is None:
            p0 = self.moments()
        XY = np.array([self.XY[0].ravel(), self.XY[1].ravel()])
        Z = self.im.ravel()
        popt, pcov, infodict, _, _ = opt.leastsq(
            errfunc, p0, args=(XY, Z), full_output=1
        )
        perr = np.sqrt(np.diag(pcov))
        self.popt = popt
        self.perr = perr
        return popt, perr


def get_meta(root, fname):
    # dir = yyyymmdd_experiment_name/p1_p1val-p2_p2val_OtherAnotations
    # fname = pname_pval.tiff
    meta = {'notes':[]}
    path = os.path.join(root, fname)
    name = os.path.splitext(fname)[0].split("_")
    scanname = "_".join(name[:-2])
    scanval, sensor = name[-2:]

    date_subdir, subsubdir = root.split(os.path.sep)[-2:]
    date = date_subdir.split("_")[-1]

    for pname_pval in subsubdir.split("-") :
        try:
            pname, pval = pname_pval.split('_')
            meta[pname] = float(pval)
        except:
            meta['notes'].append(pname_pval)

    meta["scanname"] = scanname
    meta[scanname] = float(scanval)
    meta['date'] = date
    return meta

#der = "/home/lhillber/documents/research/atom_source/experiment_data/20190412_temp_vel_check"
der = "/home/lhillber/documents/research/atom_source/tmps/experiment/nokick_MOT_free-expansion"

def add_make(k, v, dic):
    try:
        dic[k].append(v)
    except KeyError:
        dic[k] = [v]
    return dic

if __name__ == "__main__":
    fig, axs = plt.subplots(3,2, figsize=(10, 8))
    for scandir in os.listdir(der):
        subpath = os.path.join(der, scandir)
        if not os.path.isdir(subpath):
            continue

        data = {}
        print("Descending into {}".format(scandir))
        for fname in os.listdir(subpath):
            meta = get_meta(subpath, fname)
            print(meta[meta["scanname"]])
            if meta[meta["scanname"]] == 100.0:

                continue
            else:
                meta[meta["scanname"]] -= 100.0
            path = os.path.join(subpath, fname)
            print(meta)
            im = Image(path)
            im.crop(1.0, 3.0, 1.0, 3.0, units="cm")
            im.fit_crop(N=5)
            #im.show(contours=True)
            popt, perr = im.leastsq_fit()
            popt = im.moments()
            tot = np.sum(im.im)

            if meta[meta["scanname"]] >= 0.0:
                data = add_make("tot", tot/1e6, data)

                for k, v in meta.items():
                    data = add_make(k, v, data)

                for p, n, dn in zip(("x0", "y0", "sx", "sy", "theta", "A", "B"),
                    popt, perr):
                        data = add_make(p, n, data)
                        data = add_make("d"+p, n, data)

        delay = data[data["scanname"][0]]

        #if data['HH1'][0] > 0.0:
        #    label = "HH1 = {}".format(data["HH1"][0])
        #elif data["HH2"][0] > 0.0:
        #    label = "HH2 = {}".format(data["HH2"][0])
        #else:
        label = "free"
        free_data = data
        axs[0, 0].scatter(delay, data["y0"], label=label)
        axs[0, 0].set_ylabel("y0")

        axs[1, 0].scatter(delay, data["sy"])
        axs[1, 0].set_xlabel("delay")
        axs[1, 0].set_ylabel("sy")

        axs[0, 1].scatter(delay, data["x0"])
        axs[0, 1].set_ylabel("x0")

        axs[1, 1].scatter(delay, data["sx"])
        axs[1, 1].set_xlabel("delay")
        axs[1, 1].set_ylabel("sx")

        th = np.array(data["theta"])
        axs[2, 0].scatter(delay, th)
        axs[2, 0].set_xlabel("delay")
        axs[2, 0].set_ylabel("theta")


        axs[2, 1].scatter(delay, data["tot"])
        axs[2, 1].set_xlabel("delay")
        axs[2, 1].set_ylabel("tot/1e6")

        axs[0, 0].legend(ncol=2, bbox_to_anchor=(0.08, 1.1))

    plt.tight_layout()
    #plt.tight_layout(wspace=0.2, vspace=0.2, top=0.9)


    delay = np.array(free_data[free_data["scanname"][0]])
    ds = np.linspace(np.min(delay), np.max(delay), 100)
    sx2 = np.array(free_data["sx"])**2
    sy2 = np.array(free_data["sy"])**2
    # Boltzmann's constant, kg m^2 s^-2 K^-1 cm^2/m^2 s^2/us^2 K/mk

    from scipy.optimize import curve_fit

    # Boltzmann's constant, kg m^2 s^-2 K^-1 cm^2/m^2 s^2/us^2 K/mk
    kB = 1.381e-23 * 1e4 * 1e-12 * 1e-3
    # Li mass
    m = 1.16e-26

    def Tfit(t, T, s02):
        return s02 + t**2 * kB * T / m

    Txpopt, Txpcov = curve_fit(Tfit, delay, sx2)
    print("Tx = ", Txpopt[0])
    axs[1,1].plot(ds, np.sqrt(Tfit(ds, *Txpopt)), c='k')

    Typopt, Typcov = curve_fit(Tfit, delay, sy2)
    print("Ty = ", Typopt[0])
    axs[1,0].plot(ds, np.sqrt(Tfit(ds, *Typopt)), c='k')

    plt.savefig(os.path.join(der, "nokick_scan_results.pdf"))


    def filter_collections(collections, keep_dict):
        filtered_collections = collections
        for k, v in keep_dict.items():
            filtered_collections = [c for c in filtered_collections if c[k] == v]
        return filtered_collections

