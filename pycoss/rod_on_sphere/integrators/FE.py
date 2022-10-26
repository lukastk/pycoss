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

def integrator_KE_FE(c_X, t, dt, compute_N, path_handler, lmbd, taylor_tol, pre_transform=None, post_transform=None):
    f_buffer = path_handler.f_buffer
    c_shape = path_handler.c_shape
    f_shape = path_handler.f_shape

    X, N, dt_N= get_buffers(f_shape, 6)
    
    X = path_handler.iDT(c_X, out=X)

    if not pre_transform is None:
        X = pre_transform(X)

    ### Kinematics

    N = compute_N(t, X, out=N)
    
    dt_X = cov_deriv(X, N, path_handler, out=dt_X)
    X += dt*dt_X

    if not post_transform is None:
        X = post_transform(X)
    
    ### Project to mode basis

    X = path_handler.DT(X, out=c_X)

    clear_buffer(f_shape, 6)

    return X, N

def integrator_OD_FE(c_X, t, dt, compute_Q, path_handler, lmbd, taylor_tol, pre_transform=None, post_transform=None,
            compute_eF=None, compute_eM=None):
    f_buffer = path_handler.f_buffer
    c_shape = path_handler.c_shape
    f_shape = path_handler.f_shape

    X, Q, N, dt_X = get_buffers(f_shape, 4)
    
    X = path_handler.iDT(c_X, out=X)

    if not pre_transform is None:
        X = pre_transform(X)
    
    ### Dynamics
    
    Q = compute_Q(t, X, out=Q)
    N = cov_deriv(X, Q, path_handler, out=N)* (1/lmbd)

    ### Kinematics
    
    dt_X = cov_deriv(X, N, path_handler, out=dt_X)
    X += dt*dt_X

    if not post_transform is None:
        X = post_transform(X)
    
    ### Project to mode basis

    c_X = path_handler.DT(X, out=c_X)

    clear_buffer(f_shape, 4)

    return X, N, Q

def integrator_UD_FE(c_th, c_pi, c_V, c_Omg, t, dt, compute_F, compute_M, path_handler, alpha, mI, imI, lmbd, taylor_tol, pre_transform=None, post_transform=None):
    f_buffer = path_handler.f_buffer
    c_shape = path_handler.c_shape
    f_shape = path_handler.f_shape

    c_dt_th, c_dt_pi, c_dt_V, c_dt_Omg = get_buffers(c_shape, 4, dtype=path_handler.c_type)
    th, pi, V, Omg, L, dt_th, dt_pi, dt_V, dt_L, F, M = get_buffers(f_shape, 11)
    
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

    dt_V = cov_deriv(pi, F, path_handler, out=dt_V)*(1/alpha)
    f_buffer = cross(V, Omg, out=f_buffer)
    dt_V += f_buffer
    dt_V -= V*(1/alpha)

    dt_L = cov_deriv(pi, M, path_handler, out=dt_L)*(1/alpha)
    f_buffer = cross(th, F, out=f_buffer)*(1/alpha)
    dt_L += f_buffer
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
    clear_buffer(f_shape, 11)

    return th, pi, V, Omg, L, F, M
