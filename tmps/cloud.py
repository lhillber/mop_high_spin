# cloud.py
from units import kB, h

import sys
import numpy as np
from scipy.stats import skew
from numpy import pi
from numpy.linalg import norm
import matplotlib.pyplot as plt
from fastkde import fastKDE
from scipy.special import erfinv
from scipy.linalg import solve_toeplitz
from scipy.interpolate import RegularGridInterpolator
from magnetics import current_pulse
from units import muB, g
gdef = g

class Cloud:
    def __init__(
        self,
        N=1e4,
        T=100e-6,
        R=0,
        V=0,
        S=1,
        J=1/2,
        gJ=1,
        g=None,
        mass=1e-26
    ):

        self.T0 = self.to_3vec(T)
        self.S0 = self.to_3vec(S)
        self.R0 = self.to_3vec(R)
        self.V0 = self.to_3vec(V)
        self.N = int(N)
        self.J = float(J)
        self.mJ_states = np.arange(-J, J+1)
        self.gJ = gJ
        self.mass = mass
        if g is None:
            self.g = gdef
        else:
            self.g = g
        self.state = np.zeros((self.N, 7))
        self.t = 0.0

    def to_3vec(self, A):
        if type(A) in (int, float, np.float64):
            return np.array([A, A, A])
        else:
            return np.array(A)

    @property
    def xs(self):
        return self.state[::, 0:3]

    @xs.setter
    def xs(self, vals):
        self.state[::, 0:3] = vals

    @property
    def vs(self):
        return self.state[::, 3:6]

    @vs.setter
    def vs(self, vals):
        self.state[::, 3:6] = vals

    @property
    def mJs(self):
        return self.state[::, 6]

    @mJs.setter
    def mJs(self, vals):
        self.state[::, 6] = vals


    @property
    def S(self):
        return np.std(self.xs, axis=0)

    @property
    def T(self):
        c = self.mass / kB
        return c * np.var(self.vs, axis=0)

    @property
    def R(self):
        return np.mean(self.xs, axis=0)

    @property
    def V(self):
        return np.mean(self.vs, axis=0)

    @property
    def K(self):
        return skew(self.xs, axis=0)


    def initialize_state(self, spatial_distribution="gauss"):
        m = self.mass
        for i in range(3):
            v = (kB * self.T0[i] / m) ** (1 / 2)
            if spatial_distribution == "gauss":
                x = np.random.normal(self.R0[i], self.S0[i], self.N)
            elif spatial_distribution == "flat":
                x = np.random.uniform(
                    self.R0[i] - self.S0[i],
                    self.R0[i] + self.S0[i],
                    self.N)
            self.xs[:,i] = x
            self.vs[:,i] = np.random.normal(self.V0[i], v, self.N)
            self.mJs = np.random.choice(self.mJ_states, self.N)


    def set_mJs(self, coord="z", rule="equal_number"):
        if coord is not None:
            coordi = ["x","y","z", "vx","vy","vz"].index(coord)
            vals = self.state[:, coordi]
            if self.J == 1/2:
                edges = [-np.inf, self.R[coordi], np.inf]
            else:
                if rule == "equal_number":
                    matcol = np.zeros(int(2*self.J))
                    vec = np.zeros(int(2*self.J))
                    matcol[0] = 2
                    matcol[1] = -1
                    vec[0] = -1
                    vec[-1] = 1
                    sol_vec = solve_toeplitz(matcol, vec)
                    edges = [np.sqrt(2)*np.std(vals) * erfinv(E) for E in sol_vec]
                    edges = np.r_[-np.inf, edges, np.inf]
                elif rule == "equal_length":
                    ub = np.mean(vals) + 2*np.std(vals)
                    lb = np.mean(vals) - 2*np.std(vals)
                    bin_size = (ub - lb)/(2*self.J+1)
                    edges = np.arange(lb, ub+bin_size, bin_size)
                    edges[0] = -np.inf
                    edges[-1] = np.inf
            for mJi, mJ in enumerate(self.mJ_states):
                edges[mJi+1] - edges[mJi]
                mask = np.logical_and(vals>=edges[mJi], vals<edges[mJi+1])
                self.mJs[mask] = mJ


    def make_acceleration(self, pulse):
        dBdx_interp, dBdy_interp, dBdz_interp = pulse["field"].gradnormB_interp
        def a(xs, t):
            current = current_pulse(float(t), **pulse)
            coefficient = -self.mJs * self.gJ * muB / self.mass
            coefficient = coefficient[..., np.newaxis]
            gradnorm_B = np.c_[dBdx_interp(xs), dBdy_interp(xs), dBdz_interp(xs)]
            a_xyz = coefficient * current * gradnorm_B
            a_xyz[:,-1] -= self.g
            return a_xyz
        return a


    # time evolution
    def rk4(self, a):
        k1 = self.dt * a(self.xs, self.t)
        if np.sum(k1) == 0:
            self.free_expand(self.dt)
            return
        l1 = self.dt * self.vs
        k2 = self.dt * a(self.xs + l1 / 2, self.t + self.dt / 2)
        l2 = self.dt * (self.vs + k1 / 2)
        k3 = self.dt * a(self.xs + l2 / 2, self.t + self.dt / 2)
        l3 = self.dt * (self.vs + k2 / 2)
        k4 = self.dt * a(self.xs + l3, self.t + self.dt)
        l4 = self.dt * (self.vs + k3)
        self.xs = self.xs + 1 / 6 * (l1 + 2 * l2 + 2 * l3 + l4)
        self.vs = self.vs + 1 / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        self.t += self.dt
        return


    def initialize_run(self, pulses, dt):
        self.dt = dt
        self.pulses = pulses
        t_initial = self.t
        ts = []
        for pulsei, pulse in enumerate(pulses):
            if pulse["field"] is None:
                t_final = t_initial + pulse["t0"]
                pulse_ts = t_final
            else:
                t_final = t_initial + pulse["t0"] + pulse["tau"]
                pulse_ts = np.arange(t_initial, t_final, dt)
                pulse["Npts"] = len(np.arange(0, pulse["t0"] + pulse["tau"], dt))
                pulse["t0"] = t_initial
            ts = np.r_[ts, pulse_ts]
            t_initial = t_final
        ts = np.r_[ts, ts[-1]+dt]
        self.ts = ts
        self.states = np.zeros((len(ts), self.N, 7))


    def run(self):
        self.states[0] = self.state
        t_initial = self.t
        ti = 1
        for pulse in self.pulses:
            if pulse["field"] is None:
                self.free_expand(pulse["t0"])
                self.states[ti] = self.state
                ti += 1
            else:
                self.set_mJs(pulse["mJ_coord"], pulse["mJ_rule"])
                a = self.make_acceleration(pulse)
                for _ in range(pulse["Npts"]):
                    self.rk4(a)
                    self.states[ti] = self.state
                    ti += 1


    # time evolution (no forces)
    def free_expand(self, Dt):
        self.xs += self.vs * Dt
        self.xs[:, -1] -= 0.5 * self.g * Dt**2
        self.vs[:, -1] -= self.g * Dt
        self.t += Dt
        return

    # TODO enable and check recoils
    # def recoil(self):
    #    recoils = self.maxwell_velocity(370.47e-6, nc=3 * self.N)
    #    recoils = recoils.reshape(self.N, 3)
    #    self.vs = self.vs + recoils

    def plot_phasespace_all(
        self, axs=None, remove_mean=False, Nsample=10000
    ):
        print("Plotting phase space projections...")
        coord_names = {"x": "$x$",
                       "y": "$y$",
                       "z": "$z$",
                       "vx": "$v_x$",
                       "vy": "$v_y$",
                       "vz": "$v_z$"}
        if axs is None:
            fig, axs = plt.subplots(4, 4, figsize=(7, 7.5), sharex="row", sharey="col")
        else:
            fig = plt.figure()
        coords = [
            [(None, None), ("y", "x"), ("z", "y"), ("x", "z")],
            [("vx", "vy"), ("vx", "x"), ("vx", "y"), ("vx", "z")],
            [("vy", "vz"), ("vy", "x"), ("vy", "y"), ("vy", "z")],
            [("vz", "vx"), ("vz", "x"), ("vz", "y"), ("vz", "z")],
        ]
        for i in range(4):
            for j in range(4):
                ax = axs[i, j]
                ax.set_aspect("auto")
                coordx, coordy = coords[i][j]
                if (coordx, coordy) == (None, None):
                    ax.axis("off")
                    continue
                self.plot_phasespace_slice(coordx, coordy,
                    remove_mean=remove_mean, Nsample=Nsample, ax=ax)
                xname = coord_names[coordx]
                yname = coord_names[coordy]
                if i == 0:
                    ax.set_ylabel(yname)
                if i == 3:
                    ax.set_xlabel(xname)
                if j == 0:
                    ax.set_xlabel(xname)
                    ax.set_ylabel(yname)
                if i != 3:
                    plt.setp(ax.get_xticklabels(), visible=False)
                if j != 0:
                    plt.setp(ax.get_yticklabels(), visible=False)
        fig = plt.gcf()
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        return fig, axs

    def plot_phasespace_slice(self, coord_x, coord_y, ti=None, mJ_mask=None,
        ax=None, remove_mean=False, Nsample=None, scale_velocity=False,
        scatter_kwargs={}, contour_kwargs={}, contour=True, imshow=False, scatter=True):
        if ax is None:
            fig, ax = plt.subplots(1,1)
        else:
            fig = plt.gcf()
        coord_xi = ["x","y","z","vx","vy", "vz"].index(coord_x)
        coord_yi = ["x","y","z","vx","vy", "vz"].index(coord_y)
        if ti is None:
            x = self.state[:, coord_xi]
            y = self.state[:, coord_yi]
            mJs = self.mJs
        else:
            x = self.states[ti, :, coord_xi]
            y = self.states[ti, :, coord_yi]
            mJs = self.states[ti, :, -1]
        if mJ_mask is None:
            mask = np.ones_like(x, dtype=bool)
        else:
            mask = mJs == mJ_mask
        if Nsample is None:
            Nsample = self.N
        else:
            Nsample = int(Nsample)
        # convert velocities to cm/s from cm/us
        if coord_xi>=3:
            if scale_velocity:
                x = x / np.sqrt(kB*self.T0[coord_xi%3]/self.mass)
            else:
                x = x*1e6
        if coord_yi>=3:
            if scale_velocity:
                y = y / np.sqrt(kB*self.T0[coord_yi%3]/self.mass)
            else:
                y = y*1e6
        xm = np.mean(x)
        ym = np.mean(y)
        dx = 3 * np.std(x)
        dy = 3 * np.std(y)
        if remove_mean:
            x = x - xm
            y = y - ym
            xm = 0.0
            ym = 0.0

        ax.set_xlim(xm - dx, xm + dx)
        ax.set_ylim(ym - dy, ym + dy)
        Z, [xax, yax] = fastKDE.pdf(x, y)
        if imshow:
            obj = ax.imshow(Z,
                extent=[xax[0], xax[-1], yax[0], yax[-1]],
                origin="lower", aspect="equal")
        if scatter:
            interp = RegularGridInterpolator((xax, yax), Z.T)
            density = interp(np.c_[x[:Nsample], y[:Nsample]])
            idx = density.argsort()
            x, y, density, mask = x[idx], y[idx], density[idx], mask[idx]
            use_scatter_kwargs = {"s":20, "ec":"none"}
            use_scatter_kwargs.update(scatter_kwargs)
            obj = ax.scatter(x[mask], y[mask], c=density[mask], **use_scatter_kwargs)
        if contour:
            use_contour_kwargs = {"levels": 4, "colors": "k"}
            use_contour_kwargs.update(contour_kwargs)
            ax.contour(xax, yax, Z, **use_contour_kwargs)
        return obj


if __name__ == "__main__":
    # atom = Atom()

    cloud_params = dict(
        N=100,
        S=[0.25, 0.25, 0.25],
        T=[0.3, 0.3, 0.3],
        R=[0.0, 0.0, 0.0],
        V=[200 * 1e2 * 1e-6, 0.0, 0.0],
        Natoms=1e9,
        F0="thermal",
        mF0="thermal",
        constraints=[Pinhole([10.0, 0.0, 0.0], [1, 0, 0], 0.5)],
    )

    # Recalc
    cloud = Cloud(**cloud_params, recalc=True)
    print()
    cloud = Cloud(**cloud_params, recalc=False)
    print()

    print([k for k in cloud.__dict__])
    print("N", cloud.N)
    print("K", cloud.n)
    print("S", cloud.S)
    print("T", cloud.T)
    print("R", cloud.R)
    print("V", cloud.V)
    print("n", cloud.n)
    print("rho", cloud.rho)
    cloud.atom.plot_spectrum()
    cloud.plot_phasespace()
    plt.show()
