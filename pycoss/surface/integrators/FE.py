from cgi import print_environ_usage
from re import A
import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

from pycoss.surface.helpers.geometry import *

def integrator_KE_FE(thu, thv, piu, piv, t, dt, compute_V, compute_Omg, path_handler, taylor_tol, pre_transform=None, post_transform=None):
    f_buffer = path_handler.f_buffer
    f_shape = path_handler.f_shape

    V, Omg, dt_thu, dt_thv, dt_piu, dt_piv = get_buffers(f_shape, 6)

    if not pre_transform is None:
        thu, thv, piu, piv = pre_transform(thu, thv, piu, piv)

    ### Kinematics

    V = compute_V(t, thu, thv, piu, piv, out=V)
    Omg = compute_Omg(t, thu, thv, piu, piv, out=Omg)
    
    dt_thu = cov_derivu(piu, V, path_handler, out=dt_thu)
    f_buffer = cross(thu, Omg, out=f_buffer)
    dt_thu[:] += f_buffer

    dt_thv = cov_derivv(piv, V, path_handler, out=dt_thv)
    f_buffer = cross(thv, Omg, out=f_buffer)
    dt_thv[:] += f_buffer
    
    dt_piu = cov_derivu(piu, Omg, path_handler, out=dt_piu)
    dt_piv = cov_derivv(piv, Omg, path_handler, out=dt_piv)

    thu += dt*dt_thu
    thv += dt*dt_thv
    piu += dt*dt_piu
    piv += dt*dt_piv 

    if not post_transform is None:
        thu, thv, piu, piv = post_transform(thu, thv, piu, piv)
    
    ### Project to mode basis

    clear_buffer(f_shape, 6)

    return thu, thv, piu, piv, V, Omg


def integrator_OD_FE(thu, thv, piu, piv, t, dt, compute_F, compute_M, path_handler, lmbd, taylor_tol, pre_transform=None, post_transform=None,
            compute_eF=None, compute_eM=None):
    f_buffer = path_handler.f_buffer
    f_shape = path_handler.f_shape

    V, Omg, F, M, eF, eM, dt_thu, dt_thv, dt_piu, dt_piv = get_buffers(f_shape, 10)

    if not pre_transform is None:
        thu, thv, piu, piv = pre_transform(thu, thv, piu, piv)

    ### Dynamics
    
    F = compute_F(t, thu, thv, piu, piv, out=F)
    M = compute_M(t, thu, thv, piu, piv, out=M)

    if not compute_eF is None:
        eF = compute_eF(t, thu, thv, piu, piv, out=eF)
    else:
        eF[:] = 0
    
    if not compute_eM is None:
        eM = compute_eM(t, thu, thv, piu, piv, out=eM)
    else:
        eM[:] = 0

    V = cov_derivu(piu, F, path_handler, out=V)
    f_buffer = cov_derivv(piv, F, path_handler, out=f_buffer)
    V += f_buffer
    V += eF
    
    Omg = cov_derivu(piu, M, path_handler, out=Omg)
    f_buffer = cov_derivv(piv, M, path_handler, out=f_buffer)
    Omg += f_buffer
    f_buffer = cross(thu, F, out=f_buffer)
    Omg += f_buffer
    f_buffer = cross(thv, F, out=f_buffer)
    Omg += f_buffer
    Omg += eM
    Omg *= 1/lmbd

    ### Kinematics
    
    dt_thu = cov_derivu(piu, V, path_handler, out=dt_thu)
    f_buffer = cross(thu, Omg, out=f_buffer)
    dt_thu[:] += f_buffer

    dt_thv = cov_derivv(piv, V, path_handler, out=dt_thv)
    f_buffer = cross(thv, Omg, out=f_buffer)
    dt_thv[:] += f_buffer
    
    dt_piu = cov_derivu(piu, Omg, path_handler, out=dt_piu)
    dt_piv = cov_derivv(piv, Omg, path_handler, out=dt_piv)

    thu += dt*dt_thu
    thv += dt*dt_thv
    piu += dt*dt_piu
    piv += dt*dt_piv 

    if not post_transform is None:
        thu, thv, piu, piv = post_transform(thu, thv, piu, piv)
    
    ### Project to mode basis

    clear_buffer(f_shape, 10)

    return thu, thv, piu, piv, V, Omg, F, M, eF, eM

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