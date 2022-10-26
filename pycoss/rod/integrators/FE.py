from re import A
import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

from pycoss.rod.interp.FFT import *
from pycoss.rod.helpers.geometry import *


def integrator_OD_FE(c_th, c_pi, t, dt, compute_F, compute_M, path_handler, lmbd, taylor_tol, pre_transform=None, post_transform=None,
            compute_eF=None, compute_eM=None):
    f_buffer = path_handler.f_buffer
    c_shape = path_handler.c_shape
    f_shape = path_handler.f_shape

    th, pi, F, M, eF, eM, V, Omg, dt_th, dt_pi = get_buffers(f_shape, 10)
    
    th = path_handler.iDT(c_th, out=th)
    pi = path_handler.iDT(c_pi, out=pi)

    if not pre_transform is None:
        th, pi = pre_transform(th, pi)
    
    ### Dynamics
    
    F = compute_F(t, th, pi, out=F)
    M = compute_M(t, th, pi, out=M)

    if not compute_eF is None:
        eF = compute_eF(t, th, pi, out=eF)
    else:
        eF[:] = 0
    
    if not compute_eM is None:
        eM = compute_eM(t, th, pi, out=eM)
    else:
        eM[:] = 0

    V = cov_deriv(pi, F, path_handler, out=V)
    V += eF
    
    Omg = cov_deriv(pi, M, path_handler, out=Omg)
    f_buffer = cross(th, F, out=f_buffer)
    Omg += f_buffer
    Omg += eM
    Omg *= 1/lmbd

    ### Kinematics
    
    dt_th = cov_deriv(pi, V, path_handler, out=dt_th)
    f_buffer = cross(th, Omg, out=f_buffer)
    dt_th[:] += f_buffer
    
    dt_pi = cov_deriv(pi, Omg, path_handler, out=dt_pi)

    th += dt*dt_th
    pi += dt*dt_pi

    if not post_transform is None:
        th, pi = post_transform(th, pi)
    
    ### Project to mode basis

    c_th = path_handler.DT(th, out=c_th)
    c_pi = path_handler.DT(pi, out=c_pi)

    clear_buffer(f_shape, 10)

    return th, pi, V, Omg, F, M, eF, eM

def integrator_UD_FE(c_th, c_pi, c_V, c_Omg, t, dt, compute_F, compute_M, path_handler, alpha, mI, imI, lmbd, taylor_tol,
        pre_transform=None, post_transform=None, compute_eF=None, compute_eM=None):
    f_buffer = path_handler.f_buffer
    c_shape = path_handler.c_shape
    f_shape = path_handler.f_shape

    c_dt_th, c_dt_pi, c_dt_V, c_dt_Omg = get_buffers(c_shape, 4, dtype=path_handler.c_type)
    th, pi, V, Omg, L, dt_th, dt_pi, dt_V, dt_L, F, M, eF, eM = get_buffers(f_shape, 13)
    
    th = path_handler.iDT(c_th, out=th)
    pi = path_handler.iDT(c_pi, out=pi)
    V = path_handler.iDT(c_V, out=V)
    Omg = path_handler.iDT(c_Omg, out=Omg)

    if not pre_transform is None:
        th, pi, V, Omg = pre_transform(th, pi, V, Omg)

    L = np.einsum('ij,ju->iu', mI, Omg, out=L)
    
    ### Dynamics
    
    F = compute_F(t, th, pi, out=F)
    M = compute_M(t, th, pi, out=M)

    if not compute_eF is None:
        eF = compute_eF(t, th, pi, out=eF)
    else:
        eF[:] = 0
    
    if not compute_eM is None:
        eM = compute_eM(t, th, pi, out=eM)
    else:
        eM[:] = 0

    dt_V = cov_deriv(pi, F, path_handler, out=dt_V)*(1/alpha)
    f_buffer = cross(V, Omg, out=f_buffer)
    dt_V += f_buffer
    dt_V += eF
    dt_V -= V*(1/alpha)

    dt_L = cov_deriv(pi, M, path_handler, out=dt_L)*(1/alpha)
    f_buffer = cross(th, F, out=f_buffer)*(1/alpha)
    dt_L += f_buffer
    dt_L += eM
    dt_L -= Omg*(1/alpha)*lmbd
    f_buffer = cross(L, Omg, out=f_buffer)
    dt_L += f_buffer

    ### Kinematics

    dt_th = cov_deriv(pi, V, path_handler, out=dt_th)
    f_buffer = cross(th, Omg, out=f_buffer)
    dt_th += f_buffer
    
    dt_pi = cov_deriv(pi, Omg, path_handler, out=dt_pi)
    
    ### Integration

    th += dt_th*dt
    pi += dt_pi*dt
    V += dt_V*dt
    L += dt_L*dt

    Omg = np.einsum('ij,ju->iu', imI, L, out=Omg)

    if not post_transform is None:
        th, pi, V, Omg = post_transform(th, pi, V, Omg)

    ### Project to mode basis

    c_th = path_handler.DT(th, out=c_th)
    c_pi = path_handler.DT(pi, out=c_pi)
    c_V = path_handler.DT(V, out=c_V)
    c_Omg = path_handler.DT(Omg, out=c_Omg)

    clear_buffer(c_shape, 4, dtype=path_handler.c_type)
    clear_buffer(f_shape, 13)

    return th, pi, V, Omg, L, F, M

def integrator_KE_FE(c_th, c_pi, t, dt, compute_V, compute_Omg, path_handler, lmbd, taylor_tol, pre_transform=None, post_transform=None):
    f_buffer = path_handler.f_buffer
    c_shape = path_handler.c_shape
    f_shape = path_handler.f_shape

    th, pi, V, Omg, dt_th, dt_pi = get_buffers(f_shape, 6)
    
    th = path_handler.iDT(c_th, out=th)
    pi = path_handler.iDT(c_pi, out=pi)

    if not pre_transform is None:
        th, pi = pre_transform(th, pi)

    ### Kinematics

    V = compute_V(t, th, pi, out=V)
    Omg = compute_Omg(t, th, pi, out=Omg)
    
    dt_th = cov_deriv(pi, V, path_handler, out=dt_th)
    f_buffer = cross(th, Omg, out=f_buffer)
    dt_th[:] += f_buffer
    
    dt_pi = cov_deriv(pi, Omg, path_handler, out=dt_pi)

    th += dt*dt_th
    pi += dt*dt_pi

    if not post_transform is None:
        th, pi = post_transform(th, pi)
    
    ### Project to mode basis

    c_th = path_handler.DT(th, out=c_th)
    c_pi = path_handler.DT(pi, out=c_pi)

    clear_buffer(f_shape, 6)

    return th, pi, V, Omg