import numpy as np
import time
from line_profiler_pycharm import profile
from numba import jit#, prange


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


# @profile
@jit(nopython=True, fastmath=True)
def HessianNone_Fd4(CenteredFiniteDiffCoeff_dia, CenteredFiniteDiffCoeff_offdia,
                   uii1,ujj1,ukk1,
                   uij1, uik1, ujk1):
    # --------------------------------------------------------
    # ix is an np.array(3) containing the three coordinates
    # ---------------------------------------------------------

    # ---------------------------------------
    # assemble the 5x5x5 cube and convolve
    # ---------------------------------------
    # diagnoal components

    # uii =
    # ujj =
    # ukk =


    # uij =
    # uik =
    # ujk =
    # return np.dot(CenteredFiniteDiffCoeff_dia, uii1), np.dot(CenteredFiniteDiffCoeff_offdia, uij1), np.dot(CenteredFiniteDiffCoeff_offdia, uik1), np.dot(CenteredFiniteDiffCoeff_dia, ujj1), np.dot(CenteredFiniteDiffCoeff_offdia, ujk1), np.dot(CenteredFiniteDiffCoeff_dia, ukk1)
    return np.dot(CenteredFiniteDiffCoeff_offdia, uij1), np.dot(
        CenteredFiniteDiffCoeff_offdia, uik1), np.dot(CenteredFiniteDiffCoeff_offdia, ujk1)
    # return np.dot(CenteredFiniteDiffCoeff_offdia, uij1)


# @profile
@jit(nopython=True, fastmath=True, parallel=False)
def HessianFd4Lag4L(CenteredFiniteDiffCoeff_dia, CenteredFiniteDiffCoeff_offdia,
                    uii1, ujj1, ukk1,
                    uij1, uik1, ujk1):
    # --------------------------------------------------------
    # Numba doesn't work with Tensors or np.einsum/inner() etc.
    # So, we need to flatten them to vectors and use vector Numpy operations
    # Numba gives 10x+ speedup

    # ---------------------------------------
    # assemble the 4x4x4 cube and convolve
    # ---------------------------------------
    # uii = np.zeros((uShape0, 4, 4, 4))
    # ujj = np.zeros((uShape0, 4, 4, 4))
    # ukk = np.zeros((uShape0, 4, 4, 4))
    # uij = np.zeros((uShape0, 4, 4, 4))
    # uik = np.zeros((uShape0, 4, 4, 4))
    # ujk = np.zeros((uShape0, 4, 4, 4))

    uij_flat_uComp = np.zeros((4**3, 1))
    uij_flat_vComp = np.zeros((4**3, 1))
    uij_flat_wComp = np.zeros((4**3, 1))

    uik_flat_uComp = np.zeros((4**3, 1))
    uik_flat_vComp = np.zeros((4**3, 1))
    uik_flat_wComp = np.zeros((4**3, 1))

    ujk_flat_uComp = np.zeros((4**3, 1))
    ujk_flat_vComp = np.zeros((4**3, 1))
    ujk_flat_wComp = np.zeros((4**3, 1))

    for i in range(4):
        for j in range(4):
            for k in range(4):
                temp = HessianNone_Fd4(CenteredFiniteDiffCoeff_dia, CenteredFiniteDiffCoeff_offdia, uii1,ujj1,ukk1, uij1, uik1, ujk1)
                # [0] needed to convert (1x3) to (3,)
                temp_uij = temp[0][0]
                temp_uik = temp[1][0]
                temp_ujk = temp[2][0]

                uij_flat_uComp[16*i+4*j+k] = temp_uij[0]
                uij_flat_vComp[16*i+4*j+k] = temp_uij[1]
                uij_flat_wComp[16*i+4*j+k] = temp_uij[2]

                uik_flat_uComp[16*i+4*j+k] = temp_uik[0]
                uik_flat_vComp[16*i+4*j+k] = temp_uik[1]
                uik_flat_wComp[16*i+4*j+k] = temp_uik[2]

                ujk_flat_uComp[16*i+4*j+k] = temp_ujk[0]
                ujk_flat_vComp[16*i+4*j+k] = temp_ujk[1]
                ujk_flat_wComp[16*i+4*j+k] = temp_ujk[2]

    return uij_flat_uComp, uij_flat_vComp, uij_flat_wComp, uik_flat_uComp, uik_flat_vComp, uik_flat_wComp, ujk_flat_uComp, ujk_flat_vComp, ujk_flat_wComp
    # return uii, uij, uik, ujj, ujk, ukk



# @profile
def main_fn():
    for i in range(5): # repeat the experiment 5x
        dx = 2 * np.pi / 8192
        p = np.array([8, 8, 8])
        NB = 100000  # similar with LaginterpLag8C with NB = 1000000
        LW_Lag = Lag_looKuptable_4(NB)

        npoints = 5000
        # Buckets = []
        # for i in range(npoints):
        #     Buckets.append(np.random.uniform(0, 1, size=(3, 16, 16, 16)))

        Buckets = np.random.uniform(0, 1, size=(npoints, 3, 16, 16, 16))

        Hessian = []

        t1 = time.perf_counter()
        for Bucket in Buckets:
            u = Bucket
            ix = p.astype(np.int64)

            uii1 = np.ascontiguousarray(u[:, ix[0] - 2:ix[0] + 3, ix[1], ix[2]].T)
            ujj1 = np.ascontiguousarray(u[:, ix[0], ix[1] - 2:ix[1] + 3, ix[2]].T)
            ukk1 = np.ascontiguousarray(u[:, ix[0], ix[1], ix[2] - 2:ix[2] + 3].T)
            uij1 = np.array([u[:, ix[0] + 2, ix[1] + 2, ix[2]], u[:, ix[0] + 2, ix[1] - 2, ix[2]],
                             u[:, ix[0] - 2, ix[1] - 2, ix[2]],
                             u[:, ix[0] - 2, ix[1] + 2, ix[2]],
                             u[:, ix[0] + 1, ix[1] + 1, ix[2]], u[:, ix[0] + 1, ix[1] - 1, ix[2]],
                             u[:, ix[0] - 1, ix[1] - 1, ix[2]],
                             u[:, ix[0] - 1, ix[1] + 1, ix[2]]])
            uik1 = np.array(
                [u[:, ix[0] + 2, ix[1], ix[2] + 2], u[:, ix[0] + 2, ix[1], ix[2] - 2],
                 u[:, ix[0] - 2, ix[1], ix[2] - 2],
                 u[:, ix[0] - 2, ix[1], ix[2] + 2],
                 u[:, ix[0] + 1, ix[1], ix[2] + 1], u[:, ix[0] + 1, ix[1], ix[2] - 1],
                 u[:, ix[0] - 1, ix[1], ix[2] - 1],
                 u[:, ix[0] - 1, ix[1], ix[2] + 1]])
            ujk1 = np.array([u[:, ix[0], ix[1] + 2, ix[2] + 2], u[:, ix[0], ix[1] + 2, ix[2] - 2],
                             u[:, ix[0], ix[1] - 2, ix[2] - 2],
                             u[:, ix[0], ix[1] - 2, ix[2] + 2],
                             u[:, ix[0], ix[1] + 1, ix[2] + 1], u[:, ix[0], ix[1] + 1, ix[2] - 1],
                             u[:, ix[0], ix[1] - 1, ix[2] - 1],
                             u[:, ix[0], ix[1] - 1, ix[2] + 1]])
            # Hessian.append(HessianFd4Lag4L(p, Bucket, dx, LW_Lag, NB))

            CenteredFiniteDiffCoeff_dia = np.array(getNone_Fd4_diagonal(dx))
            CenteredFiniteDiffCoeff_offdia = np.array(getNone_Fd4_offdiagonal(dx))

            uij_uComp, uij_vComp, uij_wComp, uik_uComp, uik_vComp, uik_wComp, ujk_uComp, ujk_vComp, ujk_wComp = HessianFd4Lag4L(CenteredFiniteDiffCoeff_dia, CenteredFiniteDiffCoeff_offdia,
                                          uii1, ujj1, ukk1,
                                          uij1, uik1, ujk1)

            uij_uComp = uij_uComp.reshape((4,4,4))
            uij_vComp = uij_vComp.reshape((4, 4, 4))
            uij_wComp = uij_wComp.reshape((4, 4, 4))

            uik_uComp = uik_uComp.reshape((4,4,4))
            uik_vComp = uik_vComp.reshape((4, 4, 4))
            uik_wComp = uik_wComp.reshape((4, 4, 4))

            ujk_uComp = ujk_uComp.reshape((4,4,4))
            ujk_vComp = ujk_vComp.reshape((4, 4, 4))
            ujk_wComp = ujk_wComp.reshape((4, 4, 4))

            ix = p.astype(np.int64)
            fr = p - ix
            gx = LW_Lag[int(NB * fr[0])]
            gy = LW_Lag[int(NB * fr[1])]
            gz = LW_Lag[int(NB * fr[2])]
            gk = np.einsum('i,j,k', gx, gy, gz)

            uij = np.array([uij_uComp, uij_vComp, uij_wComp])
            uik = np.array([uik_uComp, uik_vComp, uik_wComp])
            ujk = np.array([ujk_uComp, ujk_vComp, ujk_wComp])

            # print(uii)
            # uii = np.einsum('ijk,lijk->l', gk, uii)  # dudxx, dvdxx, dwdxx
            # ujj = np.einsum('ijk,lijk->l', gk, ujj)  # dudyy, dvdyy, dwdyy
            # ukk = np.einsum('ijk,lijk->l', gk, ukk)  # dudzz, dvdzz, dwdzz

            uij = np.einsum('ijk,lijk->l', gk, uij)  # dudxy, dvdxy, dwdxy
            uik = np.einsum('ijk,lijk->l', gk, uik)  # dudxz, dvdxz, dwdxz
            ujk = np.einsum('ijk,lijk->l', gk, ujk)  # dudyz, dvdyz, dwdyz


            Hessian.append(uij)

        t2 = time.perf_counter()

        print('In total', (t2 - t1), ' sec')


if __name__ == '__main__':
    main_fn()
