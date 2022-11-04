import numpy as np
import time
from line_profiler_pycharm import profile


################################################################################
# Coefficients for Lagrange interpolation, 4,6,8th-order
################################################################################

def getLag4C(fr):
    # ------------------------------------------------------
    # get the 1D vectors for the 8 point Lagrange weights
    # inline the constants, and write explicit for loop
    # for the C compilation
    # ------------------------------------------------------
    # cdef int n
    wN = [1., -3., 3., -1.]
    g = np.array([0, 1., 0, 0])
    # ----------------------------
    # calculate weights if fr>0, and insert into gg
    # ----------------------------
    if (fr > 0):
        s = 0
        for n in range(4):
            g[n] = wN[n] / (fr - n + 1)
            s += g[n]
        for n in range(4):
            g[n] = g[n] / s

    return g


################################################################################
# Functions for Lagrange interpolation, 4,6,8th-order
################################################################################

def interpLag4C(p, u):
    # --------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    # ---------------------------------------------------------
    # get the coefficients
    # ----------------------
    ix = p.astype('int')
    fr = p - ix
    gx = getLag4C(fr[0])
    gy = getLag4C(fr[1])
    gz = getLag4C(fr[2])
    # ------------------------------------
    # create the 3D kernel from the
    # outer product of the 1d kernels
    # ------------------------------------
    gk = np.einsum('i,j,k', gx, gy, gz)
    # ---------------------------------------
    # assemble the 4x4x4 cube and convolve
    # ---------------------------------------
    d = u[:, ix[0] - 1:ix[0] + 3, ix[1] - 1:ix[1] + 3, ix[2] - 1:ix[2] + 3]
    ui = np.einsum('ijk,lijk->l', gk, d)

    return ui


################################################################################
# Coefficients for Hessian (central differencing), 4,6,8th-order
################################################################################
def getNone_Fd4_diagonal(dx):
    CenteredFiniteDiffCoeff = [(-1.0 / 12.0 / dx / dx, 4.0 / 3.0 / dx / dx, -15.0 / 6.0 / dx / dx,
                                4.0 / 3.0 / dx / dx, -1.0 / 12.0 / dx / dx)]
    return CenteredFiniteDiffCoeff


def getNone_Fd4_offdiagonal(dx):
    CenteredFiniteDiffCoeff = [
        (-1.0 / 48.0 / dx / dx, 1.0 / 48.0 / dx / dx, -1.0 / 48.0 / dx / dx, 1.0 / 48.0 / dx / dx,
         1.0 / 3.0 / dx / dx, -1.0 / 3.0 / dx / dx, 1.0 / 3.0 / dx / dx, -1.0 / 3.0 / dx / dx)]
    return CenteredFiniteDiffCoeff


def Lag_looKuptable_4(NB):
    frac = np.linspace(0, 1 - 1 / NB, NB)
    LW = []
    for fp in frac:
        LW.append(getLag4C(fp))

    return LW


@profile
def HessianNone_Fd4(ix, u, CenteredFiniteDiffCoeff_dia, CenteredFiniteDiffCoeff_offdia):
    # --------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    # ---------------------------------------------------------
    # get the coefficients
    # ----------------------
    # ix = p.astype(np.int64)
    # CenteredFiniteDiffCoeff_dia = getNone_Fd4_diagonal(dx)
    # CenteredFiniteDiffCoeff_offdia = getNone_Fd4_offdiagonal(dx)
    # ---------------------------------------
    # assemble the 5x5x5 cube and convolve
    # ---------------------------------------
    # diagnoal components

    uii = np.dot(CenteredFiniteDiffCoeff_dia, u[:, ix[0] - 2:ix[0] + 3, ix[1], ix[2]].T)
    ujj = np.dot(CenteredFiniteDiffCoeff_dia, u[:, ix[0], ix[1] - 2:ix[1] + 3, ix[2]].T)
    ukk = np.dot(CenteredFiniteDiffCoeff_dia, u[:, ix[0], ix[1], ix[2] - 2:ix[2] + 3].T)


    uij = np.dot(CenteredFiniteDiffCoeff_offdia, np.array(
        [u[:, ix[0] + 2, ix[1] + 2, ix[2]], u[:, ix[0] + 2, ix[1] - 2, ix[2]], u[:, ix[0] - 2, ix[1] - 2, ix[2]],
         u[:, ix[0] - 2, ix[1] + 2, ix[2]],
         u[:, ix[0] + 1, ix[1] + 1, ix[2]], u[:, ix[0] + 1, ix[1] - 1, ix[2]], u[:, ix[0] - 1, ix[1] - 1, ix[2]],
         u[:, ix[0] - 1, ix[1] + 1, ix[2]]]))
    uik = np.dot(CenteredFiniteDiffCoeff_offdia, np.array(
        [u[:, ix[0] + 2, ix[1], ix[2] + 2], u[:, ix[0] + 2, ix[1], ix[2] - 2], u[:, ix[0] - 2, ix[1], ix[2] - 2],
         u[:, ix[0] - 2, ix[1], ix[2] + 2],
         u[:, ix[0] + 1, ix[1], ix[2] + 1], u[:, ix[0] + 1, ix[1], ix[2] - 1], u[:, ix[0] - 1, ix[1], ix[2] - 1],
         u[:, ix[0] - 1, ix[1], ix[2] + 1]]))
    ujk = np.dot(CenteredFiniteDiffCoeff_offdia, np.array(
        [u[:, ix[0], ix[1] + 2, ix[2] + 2], u[:, ix[0], ix[1] + 2, ix[2] - 2], u[:, ix[0], ix[1] - 2, ix[2] - 2],
         u[:, ix[0], ix[1] - 2, ix[2] + 2],
         u[:, ix[0], ix[1] + 1, ix[2] + 1], u[:, ix[0], ix[1] + 1, ix[2] - 1], u[:, ix[0], ix[1] - 1, ix[2] - 1],
         u[:, ix[0], ix[1] - 1, ix[2] + 1]]))
    return uii, uij, uik, ujj, ujk, ukk


# @profile
def HessianFd4Lag4L(p, u, dx, LW, NB):
    # --------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    # ---------------------------------------------------------
    # get the coefficients
    # ----------------------
    ix = p.astype(np.int64)
    fr = p - ix
    gx = LW[int(NB * fr[0])]
    gy = LW[int(NB * fr[1])]
    gz = LW[int(NB * fr[2])]
    # ------------------------------------
    # create the 3D kernel from the
    # outer product of the 1d kernels
    # ------------------------------------
    gk = np.einsum('i,j,k', gx, gy, gz)
    # ---------------------------------------
    # assemble the 4x4x4 cube and convolve
    # ---------------------------------------
    uii = np.zeros((u.shape[0], 4, 4, 4))
    ujj = np.zeros((u.shape[0], 4, 4, 4))
    ukk = np.zeros((u.shape[0], 4, 4, 4))
    uij = np.zeros((u.shape[0], 4, 4, 4))
    uik = np.zeros((u.shape[0], 4, 4, 4))
    ujk = np.zeros((u.shape[0], 4, 4, 4))

    CenteredFiniteDiffCoeff_dia = getNone_Fd4_diagonal(dx)
    CenteredFiniteDiffCoeff_offdia = getNone_Fd4_offdiagonal(dx)

    for i in range(4):
        for j in range(4):
            for k in range(4):
                uii[:, i, j, k], uij[:, i, j, k], uik[:, i, j, k], ujj[:, i, j, k], ujk[:, i, j, k], ukk[:, i, j, k]\
                    = HessianNone_Fd4(np.array([ix[0] - 1 + i, ix[1] - 1 + j, ix[2] - 1 + k], dtype=np.int64), u, CenteredFiniteDiffCoeff_dia, CenteredFiniteDiffCoeff_offdia)

    uii = np.einsum('ijk,lijk->l', gk, uii)  # dudxx, dvdxx, dwdxx
    ujj = np.einsum('ijk,lijk->l', gk, ujj)  # dudyy, dvdyy, dwdyy
    ukk = np.einsum('ijk,lijk->l', gk, ukk)  # dudzz, dvdzz, dwdzz

    uij = np.einsum('ijk,lijk->l', gk, uij)  # dudxy, dvdxy, dwdxy
    uik = np.einsum('ijk,lijk->l', gk, uik)  # dudxz, dvdxz, dwdxz
    ujk = np.einsum('ijk,lijk->l', gk, ujk)  # dudyz, dvdyz, dwdyz

    return uii, uij, uik, ujj, ujk, ukk

# @profile
def main_fn():
    for i in range(5): # repeat the experiment 5x
        dx = 2 * np.pi / 8192
        p = np.array([8, 8, 8])
        NB = 100000  # similar with LaginterpLag8C with NB = 1000000
        LW_Lag = Lag_looKuptable_4(NB)

        npoints = 1000
        Buckets = []
        for i in range(npoints):
            Buckets.append(np.random.uniform(0, 1, size=(3, 16, 16, 16)))

        Hessian = []

        t1 = time.perf_counter()
        for Bucket in Buckets:
            Hessian.append(HessianFd4Lag4L(p, Bucket, dx, LW_Lag, NB))

        t2 = time.perf_counter()

        print('In total', (t2 - t1), ' sec')


if __name__ == '__main__':
    main_fn()
