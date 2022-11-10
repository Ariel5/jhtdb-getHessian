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
    return np.dot(CenteredFiniteDiffCoeff_offdia, uij1)


@profile
def HessianFd4Lag4L(p, uShape0,
                    uii1, ujj1, ukk1,
                    uij1, uik1, ujk1,
                    dx, LW, NB):
    # --------------------------------------------------------
    # p is an np.array(3) containing the three coordinates
    # ---------------------------------------------------------
    # get the coefficients
    # ----------------------

    # ---------------------------------------
    # assemble the 4x4x4 cube and convolve
    # ---------------------------------------
    uii = np.zeros((uShape0, 4, 4, 4))
    ujj = np.zeros((uShape0, 4, 4, 4))
    ukk = np.zeros((uShape0, 4, 4, 4))
    uij = np.zeros((uShape0, 4, 4, 4))
    uik = np.zeros((uShape0, 4, 4, 4))
    ujk = np.zeros((uShape0, 4, 4, 4))

    CenteredFiniteDiffCoeff_dia = np.array(getNone_Fd4_diagonal(dx))
    CenteredFiniteDiffCoeff_offdia = np.array(getNone_Fd4_offdiagonal(dx))

    u_ariel = np.zeros((4**3, 1))
    v_ariel = np.zeros((4**3, 1))
    w_ariel = np.zeros((4**3, 1))

    for i in range(4):
        for j in range(4):
            for k in range(4):
                temp = HessianNone_Fd4(CenteredFiniteDiffCoeff_dia, CenteredFiniteDiffCoeff_offdia, uii1,ujj1,ukk1, uij1, uik1, ujk1)[0]
                u_ariel[16*i+4*j+k] = temp[0]
                v_ariel[16*i+4*j+k] = temp[1]
                w_ariel[16*i+4*j+k] = temp[2]

    return u_ariel, v_ariel, w_ariel
    # return uii, uij, uik, ujj, ujk, ukk



@profile
def main_fn():
    for i in range(1): # repeat the experiment 5x
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
            u_temp_hessFd4, v_temp_hessFd4, w_temp_hessFd4 = HessianFd4Lag4L(p, u.shape[0],
                                          uii1, ujj1, ukk1,
                                          uij1, uik1, ujk1,
                                          dx, LW_Lag, NB)

            u_temp_hessFd4 = u_temp_hessFd4.reshape((4,4,4))
            v_temp_hessFd4 = v_temp_hessFd4.reshape((4, 4, 4))
            w_temp_hessFd4 = w_temp_hessFd4.reshape((4, 4, 4))

            ix = p.astype(np.int64)
            fr = p - ix
            gx = LW_Lag[int(NB * fr[0])]
            gy = LW_Lag[int(NB * fr[1])]
            gz = LW_Lag[int(NB * fr[2])]
            gk = np.einsum('i,j,k', gx, gy, gz)

            uij = np.array([u_temp_hessFd4, v_temp_hessFd4, w_temp_hessFd4])

            # print(uii)
            # uii = np.einsum('ijk,lijk->l', gk, uii)  # dudxx, dvdxx, dwdxx
            # ujj = np.einsum('ijk,lijk->l', gk, ujj)  # dudyy, dvdyy, dwdyy
            # ukk = np.einsum('ijk,lijk->l', gk, ukk)  # dudzz, dvdzz, dwdzz

            uij = np.einsum('ijk,lijk->l', gk, uij)  # dudxy, dvdxy, dwdxy
            # uik = np.einsum('ijk,lijk->l', gk, uik)  # dudxz, dvdxz, dwdxz
            # ujk = np.einsum('ijk,lijk->l', gk, ujk)  # dudyz, dvdyz, dwdyz


            Hessian.append(uij)

        t2 = time.perf_counter()

        print('In total', (t2 - t1), ' sec')


if __name__ == '__main__':
    main_fn()
