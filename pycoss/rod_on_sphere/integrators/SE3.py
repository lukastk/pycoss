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


def integrator_OD_SE3(c_th, c_pi, t, dt, compute_F, compute_M, path_handler, lmbd, taylor_tol, pre_transform=None, post_transform=None):
    f_buffer = path_handler.f_buffer
    c_shape = path_handler.c_shape
    f_shape = path_handler.f_shape
    Mm = path_handler.Mm

    th, pi, F, M, V, Omg, p, q, du_V, du_Omg, _d_A_dot_N, f_buffer2  = get_buffers(f_shape, 12)
    Omg_hat, d_A, B = get_buffers((3,3,Mm), 3)
    Omg_norm, psi = get_buffers(Mm, 2)
    taylor_mask = np.zeros(Mm, dtype=bool)
    
    th = path_handler.iDT(c_th, out=th)
    pi = path_handler.iDT(c_pi, out=pi)

    if not pre_transform is None:
        th, pi = pre_transform(th, pi)
    
    ### Dynamics
    
    F = compute_F(t, th, pi, out=F)
    M = compute_M(t, th, pi, out=M)

    V = cov_deriv(pi, F, path_handler, out=V) 
    
    Omg = cov_deriv(pi, M, path_handler, out=Omg)
    f_buffer = cross(th, F, out=f_buffer)
    Omg += f_buffer
    Omg *= 1/lmbd

    ### Kinematics

    Omg_hat = hat_vec_to_mat(Omg, out=Omg_hat)
    du_Omg = path_handler.diff_f(Omg, out=du_Omg)
    du_V = path_handler.diff_f(V, out=du_V)

    Omg_norm = norm(Omg, out=Omg_norm)
    psi[:] = dt*Omg_norm
    taylor_mask[:] = psi < taylor_tol
    
    d_A = compute_exp_se3_d_A_matrix(Omg_hat, dt, w_norm=Omg_norm, psi=psi, taylor_mask=taylor_mask, out=d_A)
    B = compute_exp_so3(Omg_hat, dt, w_norm=Omg_norm, psi=psi, taylor_mask=taylor_mask, out=B)
    q = compute_exp_adj_so3_q(Omg, du_Omg, V, dt, w_norm=Omg_norm, psi=psi, taylor_mask=taylor_mask, out=q)

    f_buffer = mat_vec_dot(d_A, V, out=f_buffer)
    f_buffer2 = mat_vec_dot(B, pi, out=f_buffer2)
    p = cross(f_buffer, f_buffer2, out=p)
    
    # Propagate th

    mat_vec_dot(B, th, out=th)
    th[:] -= p + q
    mat_vec_dot(d_A, du_V, out=_d_A_dot_N)
    th[:] += _d_A_dot_N
    
    # Propagate pi

    f_buffer[:] = du_Omg

    mat_vec_dot(B, pi, out=pi)
    mat_vec_dot(d_A, f_buffer, out=_d_A_dot_N)
    pi[:] += _d_A_dot_N
    
    ### Project to mode basis

    if not post_transform is None:
        th, pi = post_transform(th, pi)
    
    c_th = path_handler.DT(th, out=c_th)
    c_pi = path_handler.DT(pi, out=c_pi)

    clear_buffer(f_shape, 12)
    clear_buffer((3,3,Mm), 3)
    clear_buffer(Mm, 2)


    return th, pi, V, Omg, F, M

def integrator_UD_SE3(c_th, c_pi, c_V, c_Omg, t, dt, compute_F, compute_M, path_handler, alpha, mI, imI, lmbd, taylor_tol, pre_transform=None, post_transform=None):
    f_buffer = path_handler.f_buffer
    c_shape = path_handler.c_shape
    f_shape = path_handler.f_shape
    Mm = path_handler.Mm

    th, pi, F, M, V, Omg, L, p, q, du_V, du_Omg, _d_A_dot_N, f_buffer2, _th, _pi  = get_buffers(f_shape, 15)
    Omg_hat, d_A, B = get_buffers((3,3,Mm), 3)
    Omg_norm, psi = get_buffers(Mm, 2)
    taylor_mask = np.zeros(Mm, dtype=bool)

    th = path_handler.iDT(c_th, out=th)
    pi = path_handler.iDT(c_pi, out=pi)
    V = path_handler.iDT(c_V, out=V)
    Omg = path_handler.iDT(c_Omg, out=Omg)

    if not pre_transform is None:
        th, pi, V, Omg = pre_transform(th, pi, V, Omg)

    L = np.einsum('ij,ju->iu', mI, Omg, out=L)
    _th[:] = th
    _pi[:] = pi


    ### Kinematics

    Omg_hat = hat_vec_to_mat(Omg, out=Omg_hat)
    du_Omg = path_handler.diff_f(Omg, out=du_Omg)
    du_V = path_handler.diff_f(V, out=du_V)

    Omg_norm = norm(Omg, out=Omg_norm)
    psi[:] = dt*Omg_norm
    taylor_mask[:] = psi < taylor_tol
    
    d_A = compute_exp_se3_d_A_matrix(Omg_hat, dt, w_norm=Omg_norm, psi=psi, taylor_mask=taylor_mask, out=d_A)
    B = compute_exp_so3(Omg_hat, dt, w_norm=Omg_norm, psi=psi, taylor_mask=taylor_mask, out=B)
    q = compute_exp_adj_so3_q(Omg, du_Omg, V, dt, w_norm=Omg_norm, psi=psi, taylor_mask=taylor_mask, out=q)

    f_buffer = mat_vec_dot(d_A, V, out=f_buffer)
    f_buffer2 = mat_vec_dot(B, pi, out=f_buffer2)
    p = cross(f_buffer, f_buffer2, out=p)
    
    # Propagate th

    mat_vec_dot(B, th, out=th)
    th[:] -= p + q
    mat_vec_dot(d_A, du_V, out=_d_A_dot_N)
    th[:] += _d_A_dot_N
    
    # Propagate pi

    f_buffer[:] = du_Omg

    mat_vec_dot(B, pi, out=pi)
    mat_vec_dot(d_A, f_buffer, out=_d_A_dot_N)
    pi[:] += _d_A_dot_N


    ### Dynamics
    
    F = compute_F(t, _th, _pi, out=F)
    M = compute_M(t, _th, _pi, out=M)

    # Propagate V
    
    f_buffer = cov_deriv(_pi, F, path_handler, out=f_buffer)
    f_buffer -= V
    f_buffer *= (1/alpha)
    
    mat_vec_dot(B, V, out=V)
    mat_vec_dot(d_A, f_buffer, out=_d_A_dot_N)
    V[:] += _d_A_dot_N

    # Propagate L
    
    f_buffer = cov_deriv(_pi, M, path_handler, out=f_buffer)
    f_buffer2 = cross(_th, F, out=f_buffer2)
    f_buffer += f_buffer2
    f_buffer -= lmbd*Omg
    f_buffer *= (1/alpha)

    mat_vec_dot(B, L, out=L)
    mat_vec_dot(d_A, f_buffer, out=_d_A_dot_N)
    L[:] += _d_A_dot_N


    ### Project to mode basis

    Omg = np.einsum('ij,ju->iu', imI, L, out=Omg)

    if not post_transform is None:
        th, pi, V, Omg = post_transform(th, pi, V, Omg)

    c_th = path_handler.DT(th, out=c_th)
    c_pi = path_handler.DT(pi, out=c_pi)
    c_V = path_handler.DT(V, out=c_V)
    c_Omg = path_handler.DT(Omg, out=c_Omg)
    
    clear_buffer(f_shape, 12)
    clear_buffer((3,3,Mm), 3)
    clear_buffer(Mm, 2)


    return th, pi, V, Omg, L, F, M

def integrator_KE_SE3(c_th, c_pi, t, dt, compute_V, compute_Omg, path_handler, lmbd, taylor_tol, pre_transform=None, post_transform=None):
    f_buffer = path_handler.f_buffer
    c_shape = path_handler.c_shape
    f_shape = path_handler.f_shape
    Mm = path_handler.Mm

    th, pi, V, Omg, p, q, du_V, du_Omg, _d_A_dot_N, f_buffer2  = get_buffers(f_shape, 10)
    Omg_hat, d_A, B = get_buffers((3,3,Mm), 3)
    Omg_norm, psi = get_buffers(Mm, 2)
    taylor_mask = np.zeros(Mm, dtype=bool)
    
    th = path_handler.iDT(c_th, out=th)
    pi = path_handler.iDT(c_pi, out=pi)

    if not pre_transform is None:
        th, pi = pre_transform(th, pi)

    ### Kinematics

    V = compute_V(t, th, pi, out=V)
    Omg = compute_Omg(t, th, pi, out=Omg)

    Omg_hat = hat_vec_to_mat(Omg, out=Omg_hat)
    du_Omg = path_handler.diff_f(Omg, out=du_Omg)
    du_V = path_handler.diff_f(V, out=du_V)

    Omg_norm = norm(Omg, out=Omg_norm)
    psi[:] = dt*Omg_norm
    taylor_mask[:] = psi < taylor_tol
    
    d_A = compute_exp_se3_d_A_matrix(Omg_hat, dt, w_norm=Omg_norm, psi=psi, taylor_mask=taylor_mask, out=d_A)
    B = compute_exp_so3(Omg_hat, dt, w_norm=Omg_norm, psi=psi, taylor_mask=taylor_mask, out=B)
    q = compute_exp_adj_so3_q(Omg, du_Omg, V, dt, w_norm=Omg_norm, psi=psi, taylor_mask=taylor_mask, out=q)

    f_buffer = mat_vec_dot(d_A, V, out=f_buffer)
    f_buffer2 = mat_vec_dot(B, pi, out=f_buffer2)
    p = cross(f_buffer, f_buffer2, out=p)
    
    # Propagate th

    mat_vec_dot(B, th, out=th)
    th[:] -= p + q
    mat_vec_dot(d_A, du_V, out=_d_A_dot_N)
    th[:] += _d_A_dot_N
    
    # Propagate pi

    f_buffer[:] = du_Omg

    mat_vec_dot(B, pi, out=pi)
    mat_vec_dot(d_A, f_buffer, out=_d_A_dot_N)
    pi[:] += _d_A_dot_N
    
    ### Project to mode basis

    if not post_transform is None:
        th, pi = post_transform(th, pi)
    
    c_th = path_handler.DT(th, out=c_th)
    c_pi = path_handler.DT(pi, out=c_pi)

    clear_buffer(f_shape, 10)
    clear_buffer((3,3,Mm), 3)
    clear_buffer(Mm, 2)

    return th, pi, V, Omg
