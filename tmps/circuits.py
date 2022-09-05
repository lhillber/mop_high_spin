# Series RLC circuit

import numpy as np
from math import pi

def dimensions(R, d, M, N, **kwargs):
    Rmin = R - d / 2
    Rmax = R + (N - 1 / 2) * d
    Rav = (Rmin + Rmax) / 2
    width = M * d
    return Rmin, Rmax, Rav, width


def resistance(Rav, d, M, N, rho=16.78e-9 * 1e2):
    return 8 * M * N * Rav * rho / (d * d)


# Units OK?
def inductance(Rav, d, M, N, u0=1.257e-6 * 1e2):
    return (M * N) ** 2 * u0 * Rav * (np.log(16 * Rav / d) - 2) * 1e-4


def capacitance(RR, LL, tau):
    return 4 * LL * tau ** 2 / ((tau * RR) ** 2 + (2 * pi * LL) ** 2)


def RLC_values(R, d, M, N, tau, **kwargs):
    _, _, Rav, _ = dimensions(R, d, M, N)
    print(Rav)
    RR = resistance(Rav, d, M, N)
    LL = inductance(Rav, d, M, N)
    CC = capacitance(RR, LL, tau)
    return RR, LL, CC


def RLC_model(RR, LL, CC):
    alpha = RR / (2 * LL)
    omega0 = 1 / np.sqrt(LL * CC)
    beta = np.sqrt(omega0 ** 2 - alpha ** 2)
    zeta = RR / 2 * np.sqrt(CC / LL)
    if zeta > 1:
        print("over damped circuit")
    return alpha, beta


def current(t, V0, tau, R, d, M, N, **kwargs):
    RR, LL, CC = RLC_values(R, d, M, N, tau)
    print(LL)
    alpha, beta = RLC_model(RR, LL, CC)
    return V0 * np.exp(-alpha * t) * np.sin(beta * t) / (beta * LL)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    M = 7
    N = 2
    d = 0.086
    R = 3.157
    tau = 250e-6
    V0 = 200
    def curr(t):
        return current(t, V0, tau, R, d, M, N)
    ts = np.linspace(0, tau, 100)
    plt.plot(curr(ts))
    plt.show()
