from re import A
import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

from pycoss.rod.interp.FFT import *

buffers_real = {}
buffers_real_in_use = {}
buffers_complex = {}
buffers_complex_in_use = {}

buffer_dtype = float

def get_buffers(shape, buffers_required, dtype=None):

    if dtype is None:
        dtype = buffer_dtype

    if dtype == buffer_dtype:
        buffers = buffers_real
        buffers_in_use = buffers_real_in_use
    elif dtype == complex:
        buffers = buffers_complex
        buffers_in_use = buffers_complex_in_use

    if not shape in buffers:
        buffers[shape] = []
        buffers_in_use[shape] = 0

    _b = buffers[shape]
    buffer_num = len(_b)

    num_buffers_in_use = buffers_in_use[shape]
    buffers_available = buffer_num - num_buffers_in_use

    if buffers_available < buffers_required:
        for i in range(buffers_required-buffers_available):
            _b.append( np.zeros(shape, dtype=dtype) )

    buffers_in_use[shape] += buffers_required

    if buffers_required == 1:
        return _b[num_buffers_in_use:num_buffers_in_use+buffers_required][0]
    else: 
        return _b[num_buffers_in_use:num_buffers_in_use+buffers_required]

def clear_buffer(shape, num, dtype=None):
    if dtype is None:
        dtype = buffer_dtype

    if dtype == buffer_dtype:
        buffers_in_use = buffers_real_in_use
    elif dtype == complex:
        buffers_in_use = buffers_complex_in_use

    buffers_in_use[shape] -= num

    if buffers_in_use[shape] < 0:
        raise Exception('Something wrong in buffers.')

def eul2rot(theta) :

    R = np.array([[np.cos(theta[1])*np.cos(theta[2]),       np.sin(theta[0])*np.sin(theta[1])*np.cos(theta[2]) - np.sin(theta[2])*np.cos(theta[0]),      np.sin(theta[1])*np.cos(theta[0])*np.cos(theta[2]) + np.sin(theta[0])*np.sin(theta[2])],
                  [np.sin(theta[2])*np.cos(theta[1]),       np.sin(theta[0])*np.sin(theta[1])*np.sin(theta[2]) + np.cos(theta[0])*np.cos(theta[2]),      np.sin(theta[1])*np.sin(theta[2])*np.cos(theta[0]) - np.sin(theta[0])*np.cos(theta[2])],
                  [-np.sin(theta[1]),                        np.sin(theta[0])*np.cos(theta[1]),                                                           np.cos(theta[0])*np.cos(theta[1])]]
                  , dtype=theta[0].dtype)

    return R

def hat_vec_to_mat(vec, out=None):
    vec = np.array(vec)

    if out is None:
        if len(vec.shape) == 2:
            out = np.zeros((3,3,vec.shape[-1]), dtype=vec.dtype)
        else:
            out = np.zeros((3,3), dtype=vec.dtype)
        
    out[0,1] = vec[2]
    out[0,2] = -vec[1]
    out[1,2] = vec[0]
    out[1,0] = -out[0,1]
    out[2,0] = -out[0,2]
    out[2,1] = -out[1,2]

    return out
    
def hat_mat_to_vec(mat, out=None):
    if out is None:
        if len(mat.shape) == 3:
            out = np.zeros((3,mat.shape[-1]), dtype=mat.dtype)
        else:
            out = np.zeros(3, dtype=mat.dtype)        

    out[2] = mat[0,1]
    out[1] = -mat[0,2]
    out[0] = mat[1,2]

    return out

def sqr_norm(vec, out=None):
    return np.einsum('iu,iu->u', vec, vec, out=out)

def norm(vec, out=None):
    return np.sqrt(sqr_norm(vec, out))

def so3_norm(w, out=None):
    if len(w.shape) == 3:
        if out is None:
            out = np.zeros(w.shape[-1], dtype=w.dtype)
        out[:] = np.sqrt(w[0,1]**2 + w[0,2]**2 + w[1,2]**2)
        return out
    else:
        return np.sqrt(w[0,1]**2 + w[0,2]**2 + w[1,2]**2)

def vec_vec_dot(vec1, vec2, out=None):
    return np.einsum('iu,iu->u', vec1, vec2, out=out)

def mat_vec_dot(mat, vec, out=None):
    return np.einsum('iju,ju->iu', mat, vec, out=out)

def mat_mat_dot(mat1, mat2, out=None):
    return np.einsum('iju,jku->iku', mat1, mat2, out=out)

def cross(a, b, out=None):
    if out is None:
        out = np.zeros(a.shape, dtype=a.dtype)

    out[0] = -a[2]*b[1] + a[1]*b[2]
    out[1] = a[2]*b[0] - a[0]*b[2]
    out[2] = -a[1]*b[0] + a[0]*b[1]
    return out

def cov_deriv(pi, th, path_handler, out=None):
    _cov_deriv_buffer = get_buffers(th.shape, 1, dtype=th.dtype)

    if out is None:
        out = np.zeros(th.shape, dtype=th.dtype)

    path_handler.diff_f(th, out=out)
    cross(pi, th, out=_cov_deriv_buffer)
    out += _cov_deriv_buffer

    clear_buffer(th.shape, 1)

    return out

def get_R_and_frame(F):
    R = F[1:,0]
    E = F[1:,1:]
    return R, E

def get_th_and_pi(X):
    th = X[1:,0]
    pi_hat = -X[1:,1:]
    return th, pi_hat

def construct_oriented_frame(R, E, out=None):
    if out is None:
        if len(R.shape) == 2:
            out = np.zeros((4,4, R.shape[-1]), dtype=R.dtype)
        else:
            out = np.zeros((4,4), dtype=R.dtype)

    out[0,0] = 1
    out[1:,0] = R
    out[1:,1:] = E
    return out

def construct_se3_elem(th, pi_hat, out=None):
    if out is None:
        if len(th.shape) == 2:
            out = np.zeros((4, 4, th.shape[-1]), dtype=th.dtype)
        else:
            out = np.zeros((4, 4), dtype=th.dtype)

    out[1:,0] = th
    out[1:,1:] = -pi_hat
    return out

def compute_exp_se3_d_A_matrix(w, d, taylor_tol=1e-2, out=None, w_norm=None, psi=None, taylor_mask=None):
    if w_norm is None:
        w_norm = so3_norm(w)

    if psi is None:
        psi = d*w_norm

    if taylor_mask is None:
        taylor_mask = psi < taylor_tol

    if len(w.shape) == 3:
        if out is None:
            out = np.zeros((3,3,w.shape[-1]), dtype=w.dtype)

        N_taylor = np.count_nonzero(taylor_mask)
        taylor_ratio = N_taylor/w.shape[-1]
        N_zeroes = np.count_nonzero(psi == 0)

        if N_zeroes > 0:
            if taylor_ratio != 1:
                out[...,~taylor_mask] = compute_exp_se3_d_A_matrix_analytic(w[...,~taylor_mask], d, w_norm=w_norm[...,~taylor_mask], psi=psi[...,~taylor_mask])
            out[...,taylor_mask] = compute_exp_se3_d_A_matrix_taylor(w[...,taylor_mask], d, w_norm=w_norm[...,taylor_mask], psi=psi[...,taylor_mask])
        else:
            if taylor_ratio > 0.5:
                compute_exp_se3_d_A_matrix_taylor(w, d, out=out, w_norm=w_norm, psi=psi)
                if taylor_ratio != 1:
                    out[...,~taylor_mask] = compute_exp_se3_d_A_matrix_analytic(w[...,~taylor_mask], d, w_norm=w_norm[...,~taylor_mask], psi=psi[...,~taylor_mask])
            else:
                compute_exp_se3_d_A_matrix_analytic(w, d, out=out, w_norm=w_norm, psi=psi)
                if taylor_ratio != 0:
                    out[...,taylor_mask] = compute_exp_se3_d_A_matrix_taylor(w[...,taylor_mask], d, w_norm=w_norm[...,taylor_mask], psi=psi[...,taylor_mask])
    else:
        if not taylor_mask:
            out = compute_exp_se3_d_A_matrix_analytic(w, d)
        else:
            out = compute_exp_se3_d_A_matrix_taylor(w, d)

    return out

def compute_exp_se3_d_A_matrix_analytic(w, d, out=None, w_norm=None, psi=None):
    if len(w.shape) != 3:
        w = w.reshape((3,3,1))
        reshape_out = True
    else:
        reshape_out = False

    if w_norm is None:
        w_norm = so3_norm(w)

    if psi is None:
        psi = d*w_norm

    if out is None:
        out = np.zeros(w.shape, dtype=w.dtype)

    _buffer_nw, _buffer_nw_sqrd = get_buffers(w.shape, 2)

    _buffer_nw[:] = w / w_norm

    np.einsum('iju,jku->iku',_buffer_nw,_buffer_nw, out=_buffer_nw_sqrd)

    out[:] = (  (1-np.cos(psi)) * _buffer_nw + (psi - np.sin(psi)) * _buffer_nw_sqrd  ) / w_norm

    out[0,0] += d
    out[1,1] += d
    out[2,2] += d

    if reshape_out:
        out = out.reshape((3,3))

    clear_buffer(w.shape, 2)

    return out

def compute_exp_se3_d_A_matrix_taylor(w, d, out=None, w_norm=None, psi=None):
    if len(w.shape) != 3:
        w = w.reshape((3,3,1))
        reshape_out = True
    else:
        reshape_out = False

    if w_norm is None:
        w_norm = so3_norm(w)

    if psi is None:
        psi = d*w_norm

    if out is None:
        out = np.zeros(w.shape, dtype=w.dtype)

    dw, dw_sqrd = get_buffers(w.shape, 2)
    dw[:] = w*d
    np.einsum('iju,jku->iku',dw,dw, out=dw_sqrd)

    pref1, pref2, pow2_psi, pow4_psi, pow6_psi = get_buffers(psi.shape, 5)
    
    np.power(psi,2, out=pow2_psi)
    np.power(pow2_psi,2,out=pow4_psi)
    np.multiply(pow2_psi, pow4_psi, out=pow6_psi)

    pref1[:] = -2.4801587301587302e-5*pow6_psi + 0.0013888888888888889*pow4_psi - 0.041666666666666664*pow2_psi + 0.5
    pref2[:] = -2.7557319223985893e-6*pow6_psi + 0.00019841269841269841*pow4_psi - 0.0083333333333333332*pow2_psi + 0.16666666666666666

    pref1 *= d
    pref2 *= d

    out[:] = pref1*dw + pref2*dw_sqrd
    out[0,0] += d
    out[1,1] += d
    out[2,2] += d

    if reshape_out:
        out = out.reshape((3,3))

    clear_buffer(w.shape, 2)
    clear_buffer(psi.shape, 5)

    return out

def compute_exp_so3(w, d, taylor_tol=1e-2, out=None, w_norm=None, psi=None, taylor_mask=None):
    if w_norm is None:
        w_norm = so3_norm(w)

    if psi is None:
        psi = d*w_norm

    if taylor_mask is None:
        taylor_mask = psi < taylor_tol

    if len(w.shape) == 3:
        if out is None:
            out = np.zeros((3,3,w.shape[-1]), dtype=w.dtype)

        N_taylor = np.count_nonzero(taylor_mask)
        taylor_ratio = N_taylor/w.shape[-1]
        N_zeroes = np.count_nonzero(psi == 0)
        
        if N_zeroes > 0:
            if taylor_ratio != 1:
                out[...,~taylor_mask] = compute_exp_so3_analytic(w[...,~taylor_mask], d, w_norm=w_norm[...,~taylor_mask], psi=psi[...,~taylor_mask])
            out[...,taylor_mask] = compute_exp_so3_taylor(w[...,taylor_mask], d, w_norm=w_norm[...,taylor_mask], psi=psi[...,taylor_mask])
        else:
            if taylor_ratio > 0.5:
                compute_exp_so3_taylor(w, d, out=out, w_norm=w_norm, psi=psi)
                if taylor_ratio != 1:
                    out[...,~taylor_mask] = compute_exp_so3_analytic(w[...,~taylor_mask], d, w_norm=w_norm[...,~taylor_mask], psi=psi[...,~taylor_mask])
            else:
                compute_exp_so3_analytic(w, d, out=out, w_norm=w_norm, psi=psi)
                if taylor_ratio != 0:
                    out[...,taylor_mask] = compute_exp_so3_taylor(w[...,taylor_mask], d, w_norm=w_norm[...,taylor_mask], psi=psi[...,taylor_mask])
    else:
        if not taylor_mask:
            out = compute_exp_so3_analytic(w, d)
        else:
            out = compute_exp_so3_taylor(w, d)

    return out

def compute_exp_so3_analytic(w, d, out=None, w_norm=None, psi=None):
    if len(w.shape) != 3:
        w = w.reshape((3,3,1))
        reshape_out = True
    else:
        reshape_out = False

    if w_norm is None:
        w_norm = so3_norm(w)

    if psi is None:
        psi = d*w_norm

    if out is None:
        out = np.zeros(w.shape, dtype=w.dtype)

    _buffer_nw, _buffer_nw_sqrd = get_buffers(w.shape, 2)

    _buffer_nw[:] = w / w_norm

    np.einsum('iju,jku->iku',_buffer_nw,_buffer_nw, out=_buffer_nw_sqrd)

    out[:] = np.sin(psi) * _buffer_nw + (1 - np.cos(psi)) * _buffer_nw_sqrd
    out[0,0] += 1
    out[1,1] += 1
    out[2,2] += 1

    if reshape_out:
        out = out.reshape((3,3))

    clear_buffer(w.shape, 2)

    return out

def compute_exp_so3_taylor(w, d, out=None, w_norm=None, psi=None):
    if len(w.shape) != 3:
        w = w.reshape((3,3,1))
        reshape_out = True
    else:
        reshape_out = False

    if w_norm is None:
        w_norm = so3_norm(w)

    if psi is None:
        psi = d*w_norm

    if out is None:
        out = np.zeros(w.shape, dtype=w.dtype)

    dw, dw_sqrd = get_buffers(w.shape, 2)
    dw[:] = w*d
    np.einsum('iju,jku->iku',dw,dw, out=dw_sqrd)

    pref1, pref2, pow2_psi, pow4_psi, pow6_psi, pow8_psi = get_buffers(psi.shape, 6)
    
    np.power(psi,2, out=pow2_psi)
    np.power(pow2_psi,2,out=pow4_psi)
    np.multiply(pow2_psi, pow4_psi, out=pow6_psi)
    np.power(pow4_psi,2,out=pow8_psi)

    pref1[:] = 2.7557319223985893e-6*pow8_psi - 0.00019841269841269841*pow6_psi + 0.0083333333333333332*pow4_psi - 0.16666666666666666*pow2_psi + 1.0
    pref2[:] = 2.7557319223985888e-7*pow8_psi - 2.4801587301587302e-5*pow6_psi + 0.0013888888888888889*pow4_psi - 0.041666666666666664*pow2_psi + 0.5

    out[:] = pref1*dw + pref2*dw_sqrd
    out[0,0] += 1
    out[1,1] += 1
    out[2,2] += 1

    if reshape_out:
        out = out.reshape((3,3))

    clear_buffer(w.shape, 2)
    clear_buffer(psi.shape, 6)

    return out

def compute_exp_se3(X, d, taylor_tol=1e-2, out=None, w_norm=None, psi=None, taylor_mask=None):
    w = X[1:,1:]

    if w_norm is None:
        w_norm = so3_norm(w)

    if psi is None:
        psi = d*w_norm

    if taylor_mask is None:
        taylor_mask = psi < taylor_tol

    if len(X.shape) == 3:
        if out is None:
            out = np.zeros((4,4,w.shape[-1]), dtype=w.dtype)

        N_taylor = np.count_nonzero(taylor_mask)
        taylor_ratio = N_taylor/w.shape[-1]

        N_zeroes = np.count_nonzero(psi == 0)

        if N_zeroes > 0:
            if taylor_ratio != 1:
                out[...,~taylor_mask] = compute_exp_se3_analytic(X[...,~taylor_mask], d, w_norm=w_norm[...,~taylor_mask], psi=psi[...,~taylor_mask])
            out[...,taylor_mask] = compute_exp_se3_taylor(X[...,taylor_mask], d, w_norm=w_norm[...,taylor_mask], psi=psi[...,taylor_mask])
        else:
            if taylor_ratio > 0.5:
                compute_exp_se3_taylor(X, d, out=out, w_norm=w_norm, psi=psi)
                if taylor_ratio != 1:
                    out[...,~taylor_mask] = compute_exp_se3_analytic(X[...,~taylor_mask], d, w_norm=w_norm[...,~taylor_mask], psi=psi[...,~taylor_mask])
            else:
                compute_exp_se3_analytic(X, d, out=out, w_norm=w_norm, psi=psi)
                if taylor_ratio != 0:
                    out[...,taylor_mask] = compute_exp_se3_taylor(X[...,taylor_mask], d, w_norm=w_norm[...,taylor_mask], psi=psi[...,taylor_mask])
    else:
        if not taylor_mask:
            out = compute_exp_se3_analytic(X, d)
        else:
            out = compute_exp_se3_taylor(X, d)

    return out

def compute_exp_se3_analytic(X, d, out=None, w_norm=None, psi=None):
    if len(X.shape) != 3:
        X = X.reshape((4,4,1))
        reshape_out = True
    else:
        reshape_out = False

    th = X[1:,0]
    pi_T = X[1:,1:]

    if w_norm is None:
        w_norm = so3_norm(pi_T)

    if psi is None:
        psi = d*w_norm

    d_A, B = get_buffers(pi_T.shape, 2)
    
    compute_exp_se3_d_A_matrix_analytic(pi_T, d, out=d_A, w_norm=w_norm, psi=psi)
    compute_exp_so3_analytic(pi_T, d, out=B, w_norm=w_norm, psi=psi)

    if out is None:
        out = np.zeros(X.shape, dtype=X.dtype)

    out[0,0] = 1
    np.einsum('iju,ju->iu', d_A, th, out=out[1:,0]) 
    out[1:,1:] = B
    
    if reshape_out:
        out = out.reshape((4,4))

    clear_buffer(pi_T.shape, 2)

    return out

def compute_exp_se3_taylor(X, d, out=None, w_norm=None, psi=None):
    if len(X.shape) != 3:
        X = X.reshape((4,4,1))
        reshape_out = True
    else:
        reshape_out = False

    th = X[1:,0]
    pi_T = X[1:,1:]

    if w_norm is None:
        w_norm = so3_norm(pi_T)

    if psi is None:
        psi = d*w_norm

    d_A, B = get_buffers(pi_T.shape, 2)
    
    compute_exp_se3_d_A_matrix_taylor(pi_T, d, out=d_A, w_norm=w_norm, psi=psi)
    compute_exp_so3_taylor(pi_T, d, out=B, w_norm=w_norm, psi=psi)

    if out is None:
        out = np.zeros(X.shape, dtype=X.dtype)

    out[0,0] = 1
    np.einsum('iju,ju->iu', d_A, th, out=out[1:,0]) 
    out[1:,1:] = B
    
    if reshape_out:
        out = out.reshape((4,4))

    clear_buffer(pi_T.shape, 2)

    return out

def compute_exp_adj_so3_q(Omg, du_Omg, V, dt, taylor_tol=1e-2, out=None, w_norm=None, psi=None, taylor_mask=None):
    if w_norm is None:
        w_norm = norm(Omg)

    if psi is None:
        psi = dt*w_norm

    if taylor_mask is None:
        taylor_mask = psi < taylor_tol

    if len(Omg.shape) == 2:
        if out is None:
            out = np.zeros(Omg.shape, dtype=Omg.dtype)

        N_taylor = np.count_nonzero(taylor_mask)
        taylor_ratio = N_taylor/out.shape[-1]
        if taylor_ratio > 0.5:
            compute_exp_adj_so3_q_taylor(Omg, du_Omg, V, dt, out=out, norm_Omg=w_norm, psi=psi)
            if taylor_ratio != 1:
                out[...,~taylor_mask] = compute_exp_adj_so3_q_analytic(Omg[...,~taylor_mask], du_Omg[...,~taylor_mask], V[...,~taylor_mask], dt, norm_Omg=w_norm[...,~taylor_mask], psi=psi[...,~taylor_mask])
        else:
            compute_exp_adj_so3_q_analytic(Omg, du_Omg, V, dt, out=out, norm_Omg=w_norm, psi=psi)
            if taylor_ratio != 0:
                out[...,taylor_mask] = compute_exp_adj_so3_q_taylor(Omg[...,~taylor_mask], du_Omg[...,~taylor_mask], V[...,~taylor_mask], dt, norm_Omg=w_norm[...,~taylor_mask], psi=psi[...,~taylor_mask])
    else:
        if not taylor_mask:
            out = compute_exp_adj_so3_q_analytic(Omg, du_Omg, V, dt)
        else:
            out = compute_exp_adj_so3_q_taylor(Omg, du_Omg, V, dt)

    return out

def compute_exp_adj_so3_q_analytic(Omg, du_Omg, V, dt, out=None, psi=None, norm_Omg=None):

    if norm_Omg is None:
        norm_Omg = norm(Omg)

    if psi is None:
        psi = dt*norm_Omg

    pow2_psi, cos_psi, sin_psi, sin_2psi, pow2_cos_psi_m1, pow2_psi_m_sin_psi, pow2_cos_psi_m1 = get_buffers(psi.shape, 7)
    nOmg, pow2_nOmg, pow3_nOmg = get_buffers(Omg.shape, 3)
    ndu_Omg, nV = get_buffers(Omg.shape, 2)

    pow2_psi[:] = psi**2

    nOmg[:] = Omg/norm_Omg
    pow2_nOmg[:] = nOmg**2
    pow3_nOmg[:] = pow2_nOmg*nOmg

    nOmg1, nOmg2, nOmg3 = nOmg
    pow2_nOmg1, pow2_nOmg2, pow2_nOmg3 = pow2_nOmg
    pow3_nOmg1, pow3_nOmg2, pow3_nOmg3 = pow3_nOmg

    # Divide du_Omg and V by norm_Omg so that we get the correct q, and also regularise the expression
    ndu_Omg[:] = du_Omg/norm_Omg
    nV[:] = V/norm_Omg

    du_Omg1, du_Omg2, du_Omg3 = ndu_Omg
    V1, V2, V3 = nV

    cos_psi[:] = np.cos(psi)
    sin_psi[:] = np.sin(psi)
    sin_2psi[:] = np.sin(2*psi)
    pow2_cos_psi_m1[:] = np.power(cos_psi - 1.0, 2)
    pow2_psi_m_sin_psi[:] = np.power(psi - sin_psi, 2)
    pow2_cos_psi_m1[:] = np.power(cos_psi - 1.0, 2)

    if out is None:
        out = np.zeros((3,V.shape[-1]), dtype=V.dtype)

    out[0] = -V1*(du_Omg1*nOmg1*pow2_nOmg2*psi*cos_psi + 2.0*du_Omg1*nOmg1*pow2_nOmg2*psi - 3.0*du_Omg1*nOmg1*pow2_nOmg2*sin_psi + du_Omg1*nOmg1*pow2_nOmg3*psi*cos_psi + 2.0*du_Omg1*nOmg1*pow2_nOmg3*psi - 3.0*du_Omg1*nOmg1*pow2_nOmg3*sin_psi + du_Omg2*nOmg1*nOmg3*psi*sin_psi + 2.0*du_Omg2*nOmg1*nOmg3*cos_psi - 2.0*du_Omg2*nOmg1*nOmg3 + du_Omg2*pow3_nOmg2*psi*cos_psi + 2.0*du_Omg2*pow3_nOmg2*psi - 3.0*du_Omg2*pow3_nOmg2*sin_psi + du_Omg2*nOmg2*pow2_nOmg3*psi*cos_psi + 2.0*du_Omg2*nOmg2*pow2_nOmg3*psi - 3.0*du_Omg2*nOmg2*pow2_nOmg3*sin_psi - du_Omg2*nOmg2*psi*cos_psi - du_Omg2*nOmg2*psi + 2.0*du_Omg2*nOmg2*sin_psi - du_Omg3*nOmg1*nOmg2*psi*sin_psi - 2.0*du_Omg3*nOmg1*nOmg2*cos_psi + 2.0*du_Omg3*nOmg1*nOmg2 + du_Omg3*pow2_nOmg2*nOmg3*psi*cos_psi + 2.0*du_Omg3*pow2_nOmg2*nOmg3*psi - 3.0*du_Omg3*pow2_nOmg2*nOmg3*sin_psi + du_Omg3*pow3_nOmg3*psi*cos_psi + 2.0*du_Omg3*pow3_nOmg3*psi - 3.0*du_Omg3*pow3_nOmg3*sin_psi - du_Omg3*nOmg3*psi*cos_psi - du_Omg3*nOmg3*psi + 2.0*du_Omg3*nOmg3*sin_psi) + 0.25*V2*(2.0*du_Omg1*nOmg1*nOmg3*pow2_cos_psi_m1 - 2.0*du_Omg1*nOmg1*nOmg3*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) - du_Omg1*nOmg2*pow2_nOmg3*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) + du_Omg1*nOmg2*(pow2_nOmg1 + pow2_nOmg3)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) - 4.0*du_Omg1*nOmg2*(psi*cos_psi - sin_psi) - 4.0*du_Omg2*nOmg1*(psi - sin_psi) - 2.0*du_Omg2*nOmg2*nOmg3*(pow2_psi + 2.0*cos_psi - 2.0) + 2.0*du_Omg3*pow2_psi - 2.0*du_Omg3*(pow2_nOmg1 + pow2_nOmg3)*(pow2_psi + 2.0*cos_psi - 2.0) - 2.0*du_Omg3*(pow2_nOmg2 + pow2_nOmg3)*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) + 2.0*pow2_nOmg1*nOmg2*pow2_psi_m_sin_psi*(du_Omg2*nOmg3 - du_Omg3*nOmg2) + nOmg1*nOmg2*(du_Omg2*nOmg2 + du_Omg3*nOmg3)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) - nOmg1*nOmg3*(du_Omg2*nOmg3 - du_Omg3*nOmg2)*(6.0*psi - 8.0*sin_psi + sin_2psi) + nOmg1*(du_Omg1*nOmg1*nOmg2 + du_Omg2*(pow2_nOmg2 + pow2_nOmg3))*(6.0*psi - 8.0*sin_psi + sin_2psi) + 2.0*nOmg2*nOmg3*pow2_psi_m_sin_psi*(du_Omg1*nOmg1*nOmg2 + du_Omg2*(pow2_nOmg2 + pow2_nOmg3)) + 2.0*nOmg3*(du_Omg2*nOmg2 + du_Omg3*nOmg3)*pow2_cos_psi_m1 + 2.0*(pow2_nOmg1 + pow2_nOmg3)*pow2_psi_m_sin_psi*(du_Omg1*nOmg1*nOmg3 + du_Omg3*(pow2_nOmg2 + pow2_nOmg3))) + 0.25*V3*(-2.0*du_Omg1*nOmg1*nOmg2*pow2_cos_psi_m1 + 2.0*du_Omg1*nOmg1*nOmg2*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) - du_Omg1*pow2_nOmg2*nOmg3*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) + du_Omg1*nOmg3*(pow2_nOmg1 + pow2_nOmg2)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) - 4.0*du_Omg1*nOmg3*(psi*cos_psi - sin_psi) - 2.0*du_Omg2*pow2_psi + 2.0*du_Omg2*(pow2_nOmg1 + pow2_nOmg2)*(pow2_psi + 2.0*cos_psi - 2.0) + 2.0*du_Omg2*(pow2_nOmg2 + pow2_nOmg3)*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) - 4.0*du_Omg3*nOmg1*(psi - sin_psi) + 2.0*du_Omg3*nOmg2*nOmg3*(pow2_psi + 2.0*cos_psi - 2.0) + 2.0*pow2_nOmg1*nOmg3*pow2_psi_m_sin_psi*(du_Omg2*nOmg3 - du_Omg3*nOmg2) + nOmg1*nOmg2*(du_Omg2*nOmg3 - du_Omg3*nOmg2)*(6.0*psi - 8.0*sin_psi + sin_2psi) + nOmg1*nOmg3*(du_Omg2*nOmg2 + du_Omg3*nOmg3)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) + nOmg1*(du_Omg1*nOmg1*nOmg3 + du_Omg3*(pow2_nOmg2 + pow2_nOmg3))*(6.0*psi - 8.0*sin_psi + sin_2psi) - 2.0*nOmg2*nOmg3*pow2_psi_m_sin_psi*(du_Omg1*nOmg1*nOmg3 + du_Omg3*(pow2_nOmg2 + pow2_nOmg3)) - 2.0*nOmg2*(du_Omg2*nOmg2 + du_Omg3*nOmg3)*pow2_cos_psi_m1 - 2.0*(pow2_nOmg1 + pow2_nOmg2)*pow2_psi_m_sin_psi*(du_Omg1*nOmg1*nOmg2 + du_Omg2*(pow2_nOmg2 + pow2_nOmg3)))
    out[1] = -0.25*V1*(-2.0*du_Omg1*nOmg1*nOmg3*(pow2_psi + 2.0*cos_psi - 2.0) + 4.0*du_Omg1*nOmg2*(psi - sin_psi) + du_Omg2*nOmg1*pow2_nOmg3*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) - du_Omg2*nOmg1*(pow2_nOmg2 + pow2_nOmg3)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) + 4.0*du_Omg2*nOmg1*(psi*cos_psi - sin_psi) + 2.0*du_Omg2*nOmg2*nOmg3*pow2_cos_psi_m1 - 2.0*du_Omg2*nOmg2*nOmg3*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) + 2.0*du_Omg3*pow2_psi - 2.0*du_Omg3*(pow2_nOmg1 + pow2_nOmg3)*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) - 2.0*du_Omg3*(pow2_nOmg2 + pow2_nOmg3)*(pow2_psi + 2.0*cos_psi - 2.0) + 2.0*nOmg1*pow2_nOmg2*pow2_psi_m_sin_psi*(du_Omg1*nOmg3 - du_Omg3*nOmg1) - nOmg1*nOmg2*(du_Omg1*nOmg1 + du_Omg3*nOmg3)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) + 2.0*nOmg1*nOmg3*pow2_psi_m_sin_psi*(du_Omg1*(pow2_nOmg1 + pow2_nOmg3) + du_Omg2*nOmg1*nOmg2) + nOmg2*nOmg3*(du_Omg1*nOmg3 - du_Omg3*nOmg1)*(6.0*psi - 8.0*sin_psi + sin_2psi) - nOmg2*(du_Omg1*(pow2_nOmg1 + pow2_nOmg3) + du_Omg2*nOmg1*nOmg2)*(6.0*psi - 8.0*sin_psi + sin_2psi) + 2.0*nOmg3*(du_Omg1*nOmg1 + du_Omg3*nOmg3)*pow2_cos_psi_m1 + 2.0*(pow2_nOmg2 + pow2_nOmg3)*pow2_psi_m_sin_psi*(du_Omg2*nOmg2*nOmg3 + du_Omg3*(pow2_nOmg1 + pow2_nOmg3))) - V2*(du_Omg1*pow3_nOmg1*psi*cos_psi + 2.0*du_Omg1*pow3_nOmg1*psi - 3.0*du_Omg1*pow3_nOmg1*sin_psi + du_Omg1*nOmg1*pow2_nOmg3*psi*cos_psi + 2.0*du_Omg1*nOmg1*pow2_nOmg3*psi - 3.0*du_Omg1*nOmg1*pow2_nOmg3*sin_psi - du_Omg1*nOmg1*psi*cos_psi - du_Omg1*nOmg1*psi + 2.0*du_Omg1*nOmg1*sin_psi - du_Omg1*nOmg2*nOmg3*psi*sin_psi - 2.0*du_Omg1*nOmg2*nOmg3*cos_psi + 2.0*du_Omg1*nOmg2*nOmg3 + du_Omg2*pow2_nOmg1*nOmg2*psi*cos_psi + 2.0*du_Omg2*pow2_nOmg1*nOmg2*psi - 3.0*du_Omg2*pow2_nOmg1*nOmg2*sin_psi + du_Omg2*nOmg2*pow2_nOmg3*psi*cos_psi + 2.0*du_Omg2*nOmg2*pow2_nOmg3*psi - 3.0*du_Omg2*nOmg2*pow2_nOmg3*sin_psi + du_Omg3*pow2_nOmg1*nOmg3*psi*cos_psi + 2.0*du_Omg3*pow2_nOmg1*nOmg3*psi - 3.0*du_Omg3*pow2_nOmg1*nOmg3*sin_psi + du_Omg3*nOmg1*nOmg2*psi*sin_psi + 2.0*du_Omg3*nOmg1*nOmg2*cos_psi - 2.0*du_Omg3*nOmg1*nOmg2 + du_Omg3*pow3_nOmg3*psi*cos_psi + 2.0*du_Omg3*pow3_nOmg3*psi - 3.0*du_Omg3*pow3_nOmg3*sin_psi - du_Omg3*nOmg3*psi*cos_psi - du_Omg3*nOmg3*psi + 2.0*du_Omg3*nOmg3*sin_psi) + 0.25*V3*(2.0*du_Omg1*pow2_psi - 2.0*du_Omg1*(pow2_nOmg1 + pow2_nOmg2)*(pow2_psi + 2.0*cos_psi - 2.0) - 2.0*du_Omg1*(pow2_nOmg1 + pow2_nOmg3)*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) - du_Omg2*pow2_nOmg1*nOmg3*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) + 2.0*du_Omg2*nOmg1*nOmg2*pow2_cos_psi_m1 - 2.0*du_Omg2*nOmg1*nOmg2*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) + du_Omg2*nOmg3*(pow2_nOmg1 + pow2_nOmg2)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) - 4.0*du_Omg2*nOmg3*(psi*cos_psi - sin_psi) - 2.0*du_Omg3*nOmg1*nOmg3*(pow2_psi + 2.0*cos_psi - 2.0) - 4.0*du_Omg3*nOmg2*(psi - sin_psi) + nOmg1*nOmg2*(du_Omg1*nOmg3 - du_Omg3*nOmg1)*(6.0*psi - 8.0*sin_psi + sin_2psi) + 2.0*nOmg1*nOmg3*pow2_psi_m_sin_psi*(du_Omg2*nOmg2*nOmg3 + du_Omg3*(pow2_nOmg1 + pow2_nOmg3)) + 2.0*nOmg1*(du_Omg1*nOmg1 + du_Omg3*nOmg3)*pow2_cos_psi_m1 - 2.0*pow2_nOmg2*nOmg3*pow2_psi_m_sin_psi*(du_Omg1*nOmg3 - du_Omg3*nOmg1) + nOmg2*nOmg3*(du_Omg1*nOmg1 + du_Omg3*nOmg3)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) + nOmg2*(du_Omg2*nOmg2*nOmg3 + du_Omg3*(pow2_nOmg1 + pow2_nOmg3))*(6.0*psi - 8.0*sin_psi + sin_2psi) + 2.0*(pow2_nOmg1 + pow2_nOmg2)*pow2_psi_m_sin_psi*(du_Omg1*(pow2_nOmg1 + pow2_nOmg3) + du_Omg2*nOmg1*nOmg2))
    out[2] = 0.25*V1*(-2.0*du_Omg1*nOmg1*nOmg2*(pow2_psi + 2.0*cos_psi - 2.0) - 4.0*du_Omg1*nOmg3*(psi - sin_psi) + 2.0*du_Omg2*pow2_psi - 2.0*du_Omg2*(pow2_nOmg1 + pow2_nOmg2)*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) - 2.0*du_Omg2*(pow2_nOmg2 + pow2_nOmg3)*(pow2_psi + 2.0*cos_psi - 2.0) - du_Omg3*nOmg1*pow2_nOmg2*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) + du_Omg3*nOmg1*(pow2_nOmg2 + pow2_nOmg3)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) - 4.0*du_Omg3*nOmg1*(psi*cos_psi - sin_psi) + 2.0*du_Omg3*nOmg2*nOmg3*pow2_cos_psi_m1 - 2.0*du_Omg3*nOmg2*nOmg3*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) + 2.0*nOmg1*nOmg2*pow2_psi_m_sin_psi*(du_Omg1*(pow2_nOmg1 + pow2_nOmg2) + du_Omg3*nOmg1*nOmg3) + 2.0*nOmg1*pow2_nOmg3*pow2_psi_m_sin_psi*(du_Omg1*nOmg2 - du_Omg2*nOmg1) + nOmg1*nOmg3*(du_Omg1*nOmg1 + du_Omg2*nOmg2)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) - nOmg2*nOmg3*(du_Omg1*nOmg2 - du_Omg2*nOmg1)*(6.0*psi - 8.0*sin_psi + sin_2psi) + 2.0*nOmg2*(du_Omg1*nOmg1 + du_Omg2*nOmg2)*pow2_cos_psi_m1 + nOmg3*(du_Omg1*(pow2_nOmg1 + pow2_nOmg2) + du_Omg3*nOmg1*nOmg3)*(6.0*psi - 8.0*sin_psi + sin_2psi) + 2.0*(pow2_nOmg2 + pow2_nOmg3)*pow2_psi_m_sin_psi*(du_Omg2*(pow2_nOmg1 + pow2_nOmg2) + du_Omg3*nOmg2*nOmg3)) + 0.25*V2*(-2.0*du_Omg1*pow2_psi + 2.0*du_Omg1*(pow2_nOmg1 + pow2_nOmg2)*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) + 2.0*du_Omg1*(pow2_nOmg1 + pow2_nOmg3)*(pow2_psi + 2.0*cos_psi - 2.0) + 2.0*du_Omg2*nOmg1*nOmg2*(pow2_psi + 2.0*cos_psi - 2.0) - 4.0*du_Omg2*nOmg3*(psi - sin_psi) - du_Omg3*pow2_nOmg1*nOmg2*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) - 2.0*du_Omg3*nOmg1*nOmg3*pow2_cos_psi_m1 + 2.0*du_Omg3*nOmg1*nOmg3*(pow2_psi - 2.0*psi*sin_psi - 2.0*cos_psi + 2.0) + du_Omg3*nOmg2*(pow2_nOmg1 + pow2_nOmg3)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) - 4.0*du_Omg3*nOmg2*(psi*cos_psi - sin_psi) - 2.0*nOmg1*nOmg2*pow2_psi_m_sin_psi*(du_Omg2*(pow2_nOmg1 + pow2_nOmg2) + du_Omg3*nOmg2*nOmg3) + nOmg1*nOmg3*(du_Omg1*nOmg2 - du_Omg2*nOmg1)*(6.0*psi - 8.0*sin_psi + sin_2psi) - 2.0*nOmg1*(du_Omg1*nOmg1 + du_Omg2*nOmg2)*pow2_cos_psi_m1 + 2.0*nOmg2*pow2_nOmg3*pow2_psi_m_sin_psi*(du_Omg1*nOmg2 - du_Omg2*nOmg1) + nOmg2*nOmg3*(du_Omg1*nOmg1 + du_Omg2*nOmg2)*(4.0*psi*cos_psi + 2.0*psi - 4.0*sin_psi - sin_2psi) + nOmg3*(du_Omg2*(pow2_nOmg1 + pow2_nOmg2) + du_Omg3*nOmg2*nOmg3)*(6.0*psi - 8.0*sin_psi + sin_2psi) - 2.0*(pow2_nOmg1 + pow2_nOmg3)*pow2_psi_m_sin_psi*(du_Omg1*(pow2_nOmg1 + pow2_nOmg2) + du_Omg3*nOmg1*nOmg3)) - V3*(du_Omg1*pow3_nOmg1*psi*cos_psi + 2.0*du_Omg1*pow3_nOmg1*psi - 3.0*du_Omg1*pow3_nOmg1*sin_psi + du_Omg1*nOmg1*pow2_nOmg2*psi*cos_psi + 2.0*du_Omg1*nOmg1*pow2_nOmg2*psi - 3.0*du_Omg1*nOmg1*pow2_nOmg2*sin_psi - du_Omg1*nOmg1*psi*cos_psi - du_Omg1*nOmg1*psi + 2.0*du_Omg1*nOmg1*sin_psi + du_Omg1*nOmg2*nOmg3*psi*sin_psi + 2.0*du_Omg1*nOmg2*nOmg3*cos_psi - 2.0*du_Omg1*nOmg2*nOmg3 + du_Omg2*pow2_nOmg1*nOmg2*psi*cos_psi + 2.0*du_Omg2*pow2_nOmg1*nOmg2*psi - 3.0*du_Omg2*pow2_nOmg1*nOmg2*sin_psi - du_Omg2*nOmg1*nOmg3*psi*sin_psi - 2.0*du_Omg2*nOmg1*nOmg3*cos_psi + 2.0*du_Omg2*nOmg1*nOmg3 + du_Omg2*pow3_nOmg2*psi*cos_psi + 2.0*du_Omg2*pow3_nOmg2*psi - 3.0*du_Omg2*pow3_nOmg2*sin_psi - du_Omg2*nOmg2*psi*cos_psi - du_Omg2*nOmg2*psi + 2.0*du_Omg2*nOmg2*sin_psi + du_Omg3*pow2_nOmg1*nOmg3*psi*cos_psi + 2.0*du_Omg3*pow2_nOmg1*nOmg3*psi - 3.0*du_Omg3*pow2_nOmg1*nOmg3*sin_psi + du_Omg3*pow2_nOmg2*nOmg3*psi*cos_psi + 2.0*du_Omg3*pow2_nOmg2*nOmg3*psi - 3.0*du_Omg3*pow2_nOmg2*nOmg3*sin_psi)

    clear_buffer(psi.shape, 7)
    clear_buffer(Omg.shape, 5)

    return out

def compute_exp_adj_so3_q_taylor(Omg, du_Omg, V, dt, out=None, psi=None, norm_Omg=None):

    if norm_Omg is None:
        norm_Omg = norm(Omg)

    if psi is None:
        psi = dt*norm_Omg

    pow2_psi, pow3_psi, pow4_psi, pow5_psi, pow6_psi = get_buffers(psi.shape, 5)
    tdOmg, pow2_tdOmg, pow3_tdOmg, pow4_tdOmg = get_buffers(Omg.shape, 4)
    du_Omg_dt, V_dt = get_buffers(Omg.shape, 2)

    pow2_psi[:] = psi**2
    pow3_psi[:] = pow2_psi*psi
    pow4_psi[:] = pow3_psi*psi
    pow5_psi[:] = pow4_psi*psi
    pow6_psi[:] = pow5_psi*psi

    tdOmg[:] = Omg*dt
    pow2_tdOmg[:] = tdOmg**2
    pow3_tdOmg[:] = pow2_tdOmg*tdOmg
    pow4_tdOmg[:] = pow2_tdOmg**2

    tdOmg1, tdOmg2, tdOmg3 = tdOmg
    pow2_tdOmg1, pow2_tdOmg2, pow2_tdOmg3 = pow2_tdOmg
    pow3_tdOmg1, pow3_tdOmg2, pow3_tdOmg3 = pow3_tdOmg
    pow4_tdOmg1, pow4_tdOmg2, pow4_tdOmg3 = pow4_tdOmg

    du_Omg_dt[:] = du_Omg*dt
    V_dt[:] = V*dt
    du_Omg1_dt, du_Omg2_dt, du_Omg3_dt = du_Omg_dt
    V1_dt, V2_dt, V3_dt = V_dt

    if out is None:
        out = np.zeros((3,V.shape[-1]), dtype=V.dtype)

    out[0] = -0.016666666666666666*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2 - 0.016666666666666666*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg3 + 0.083333333333333329*V1_dt*du_Omg2_dt*tdOmg1*tdOmg3 - 0.016666666666666666*V1_dt*du_Omg2_dt*pow3_tdOmg2 - 0.016666666666666666*V1_dt*du_Omg2_dt*tdOmg2*pow2_tdOmg3 - 0.16666666666666666*V1_dt*du_Omg2_dt*tdOmg2 - 0.083333333333333329*V1_dt*du_Omg3_dt*tdOmg1*tdOmg2 - 0.016666666666666666*V1_dt*du_Omg3_dt*pow2_tdOmg2*tdOmg3 - 0.016666666666666666*V1_dt*du_Omg3_dt*pow3_tdOmg3 - 0.16666666666666666*V1_dt*du_Omg3_dt*tdOmg3 + 0.013888888888888888*V2_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg3 + 0.016666666666666666*V2_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg2 + 0.013888888888888888*V2_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2*tdOmg3 + 0.013888888888888888*V2_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg3 + 0.33333333333333331*V2_dt*du_Omg1_dt*tdOmg2 + 0.013888888888888888*V2_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2*tdOmg3 + 0.016666666666666666*V2_dt*du_Omg2_dt*tdOmg1*pow2_tdOmg2 - 0.16666666666666666*V2_dt*du_Omg2_dt*tdOmg1 + 0.013888888888888888*V2_dt*du_Omg2_dt*pow3_tdOmg2*tdOmg3 + 0.013888888888888888*V2_dt*du_Omg2_dt*tdOmg2*pow3_tdOmg3 + 0.083333333333333329*V2_dt*du_Omg2_dt*tdOmg2*tdOmg3 + 0.013888888888888888*V2_dt*du_Omg3_dt*pow2_tdOmg1*pow2_tdOmg3 - 0.041666666666666664*V2_dt*du_Omg3_dt*pow2_tdOmg1 + 0.016666666666666666*V2_dt*du_Omg3_dt*tdOmg1*tdOmg2*tdOmg3 + 0.013888888888888888*V2_dt*du_Omg3_dt*pow2_tdOmg2*pow2_tdOmg3 - 0.125*V2_dt*du_Omg3_dt*pow2_tdOmg2 + 0.013888888888888888*V2_dt*du_Omg3_dt*pow4_tdOmg3 - 0.041666666666666664*V2_dt*du_Omg3_dt*pow2_tdOmg3 + 0.5*V2_dt*du_Omg3_dt - 0.013888888888888888*V3_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg2 + 0.016666666666666666*V3_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg3 - 0.013888888888888888*V3_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg2 - 0.013888888888888888*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2*pow2_tdOmg3 + 0.33333333333333331*V3_dt*du_Omg1_dt*tdOmg3 - 0.013888888888888888*V3_dt*du_Omg2_dt*pow2_tdOmg1*pow2_tdOmg2 + 0.041666666666666664*V3_dt*du_Omg2_dt*pow2_tdOmg1 + 0.016666666666666666*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2*tdOmg3 - 0.013888888888888888*V3_dt*du_Omg2_dt*pow4_tdOmg2 - 0.013888888888888888*V3_dt*du_Omg2_dt*pow2_tdOmg2*pow2_tdOmg3 + 0.041666666666666664*V3_dt*du_Omg2_dt*pow2_tdOmg2 + 0.125*V3_dt*du_Omg2_dt*pow2_tdOmg3 - 0.5*V3_dt*du_Omg2_dt - 0.013888888888888888*V3_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg2*tdOmg3 + 0.016666666666666666*V3_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg3 - 0.16666666666666666*V3_dt*du_Omg3_dt*tdOmg1 - 0.013888888888888888*V3_dt*du_Omg3_dt*pow3_tdOmg2*tdOmg3 - 0.013888888888888888*V3_dt*du_Omg3_dt*tdOmg2*pow3_tdOmg3 - 0.083333333333333329*V3_dt*du_Omg3_dt*tdOmg2*tdOmg3 + 8.3507027951472401e-9*pow6_psi*(24.0*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2 + 24.0*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg3 - 264.0*V1_dt*du_Omg2_dt*tdOmg1*tdOmg3 + 24.0*V1_dt*du_Omg2_dt*pow3_tdOmg2 + 24.0*V1_dt*du_Omg2_dt*tdOmg2*pow2_tdOmg3 + 2310.0*V1_dt*du_Omg2_dt*tdOmg2 + 264.0*V1_dt*du_Omg3_dt*tdOmg1*tdOmg2 + 24.0*V1_dt*du_Omg3_dt*pow2_tdOmg2*tdOmg3 + 24.0*V1_dt*du_Omg3_dt*pow3_tdOmg3 + 2310.0*V1_dt*du_Omg3_dt*tdOmg3 - 253.0*V2_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg3 - 24.0*V2_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg2 - 253.0*V2_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2*tdOmg3 - 253.0*V2_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg3 - 8118.0*V2_dt*du_Omg1_dt*tdOmg1*tdOmg3 - 2640.0*V2_dt*du_Omg1_dt*tdOmg2 - 253.0*V2_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2*tdOmg3 - 24.0*V2_dt*du_Omg2_dt*tdOmg1*pow2_tdOmg2 + 330.0*V2_dt*du_Omg2_dt*tdOmg1 - 253.0*V2_dt*du_Omg2_dt*pow3_tdOmg2*tdOmg3 - 253.0*V2_dt*du_Omg2_dt*tdOmg2*pow3_tdOmg3 - 8382.0*V2_dt*du_Omg2_dt*tdOmg2*tdOmg3 - 253.0*V2_dt*du_Omg3_dt*pow2_tdOmg1*pow2_tdOmg3 + 33.0*V2_dt*du_Omg3_dt*pow2_tdOmg1 - 24.0*V2_dt*du_Omg3_dt*tdOmg1*tdOmg2*tdOmg3 - 253.0*V2_dt*du_Omg3_dt*pow2_tdOmg2*pow2_tdOmg3 + 297.0*V2_dt*du_Omg3_dt*pow2_tdOmg2 - 253.0*V2_dt*du_Omg3_dt*pow4_tdOmg3 - 8085.0*V2_dt*du_Omg3_dt*pow2_tdOmg3 + 253.0*V3_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg2 - 24.0*V3_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg3 + 253.0*V3_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg2 + 253.0*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2*pow2_tdOmg3 + 8118.0*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2 - 2640.0*V3_dt*du_Omg1_dt*tdOmg3 + 253.0*V3_dt*du_Omg2_dt*pow2_tdOmg1*pow2_tdOmg2 - 33.0*V3_dt*du_Omg2_dt*pow2_tdOmg1 - 24.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2*tdOmg3 + 253.0*V3_dt*du_Omg2_dt*pow4_tdOmg2 + 253.0*V3_dt*du_Omg2_dt*pow2_tdOmg2*pow2_tdOmg3 + 8085.0*V3_dt*du_Omg2_dt*pow2_tdOmg2 - 297.0*V3_dt*du_Omg2_dt*pow2_tdOmg3 + 253.0*V3_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg2*tdOmg3 - 24.0*V3_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg3 + 330.0*V3_dt*du_Omg3_dt*tdOmg1 + 253.0*V3_dt*du_Omg3_dt*pow3_tdOmg2*tdOmg3 + 253.0*V3_dt*du_Omg3_dt*tdOmg2*pow3_tdOmg3 + 8382.0*V3_dt*du_Omg3_dt*tdOmg2*tdOmg3) + 1.6534391534391535e-6*pow4_psi*(-10.0*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2 - 10.0*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg3 + 90.0*V1_dt*du_Omg2_dt*tdOmg1*tdOmg3 - 10.0*V1_dt*du_Omg2_dt*pow3_tdOmg2 - 10.0*V1_dt*du_Omg2_dt*tdOmg2*pow2_tdOmg3 - 600.0*V1_dt*du_Omg2_dt*tdOmg2 - 90.0*V1_dt*du_Omg3_dt*tdOmg1*tdOmg2 - 10.0*V1_dt*du_Omg3_dt*pow2_tdOmg2*tdOmg3 - 10.0*V1_dt*du_Omg3_dt*pow3_tdOmg3 - 600.0*V1_dt*du_Omg3_dt*tdOmg3 + 41.0*V2_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg3 + 10.0*V2_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg2 + 41.0*V2_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2*tdOmg3 + 41.0*V2_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg3 + 840.0*V2_dt*du_Omg1_dt*tdOmg1*tdOmg3 + 720.0*V2_dt*du_Omg1_dt*tdOmg2 + 41.0*V2_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2*tdOmg3 + 10.0*V2_dt*du_Omg2_dt*tdOmg1*pow2_tdOmg2 - 120.0*V2_dt*du_Omg2_dt*tdOmg1 + 41.0*V2_dt*du_Omg2_dt*pow3_tdOmg2*tdOmg3 + 41.0*V2_dt*du_Omg2_dt*tdOmg2*pow3_tdOmg3 + 930.0*V2_dt*du_Omg2_dt*tdOmg2*tdOmg3 + 41.0*V2_dt*du_Omg3_dt*pow2_tdOmg1*pow2_tdOmg3 - 15.0*V2_dt*du_Omg3_dt*pow2_tdOmg1 + 10.0*V2_dt*du_Omg3_dt*tdOmg1*tdOmg2*tdOmg3 + 41.0*V2_dt*du_Omg3_dt*pow2_tdOmg2*pow2_tdOmg3 - 105.0*V2_dt*du_Omg3_dt*pow2_tdOmg2 + 41.0*V2_dt*du_Omg3_dt*pow4_tdOmg3 + 825.0*V2_dt*du_Omg3_dt*pow2_tdOmg3 - 41.0*V3_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg2 + 10.0*V3_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg3 - 41.0*V3_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg2 - 41.0*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2*pow2_tdOmg3 - 840.0*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2 + 720.0*V3_dt*du_Omg1_dt*tdOmg3 - 41.0*V3_dt*du_Omg2_dt*pow2_tdOmg1*pow2_tdOmg2 + 15.0*V3_dt*du_Omg2_dt*pow2_tdOmg1 + 10.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2*tdOmg3 - 41.0*V3_dt*du_Omg2_dt*pow4_tdOmg2 - 41.0*V3_dt*du_Omg2_dt*pow2_tdOmg2*pow2_tdOmg3 - 825.0*V3_dt*du_Omg2_dt*pow2_tdOmg2 + 105.0*V3_dt*du_Omg2_dt*pow2_tdOmg3 - 41.0*V3_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg2*tdOmg3 + 10.0*V3_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg3 - 120.0*V3_dt*du_Omg3_dt*tdOmg1 - 41.0*V3_dt*du_Omg3_dt*pow3_tdOmg2*tdOmg3 - 41.0*V3_dt*du_Omg3_dt*tdOmg2*pow3_tdOmg3 - 930.0*V3_dt*du_Omg3_dt*tdOmg2*tdOmg3) + 0.00019841269841269841*pow2_psi*(4.0*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2 + 4.0*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg3 - 28.0*V1_dt*du_Omg2_dt*tdOmg1*tdOmg3 + 4.0*V1_dt*du_Omg2_dt*pow3_tdOmg2 + 4.0*V1_dt*du_Omg2_dt*tdOmg2*pow2_tdOmg3 + 126.0*V1_dt*du_Omg2_dt*tdOmg2 + 28.0*V1_dt*du_Omg3_dt*tdOmg1*tdOmg2 + 4.0*V1_dt*du_Omg3_dt*pow2_tdOmg2*tdOmg3 + 4.0*V1_dt*du_Omg3_dt*pow3_tdOmg3 + 126.0*V1_dt*du_Omg3_dt*tdOmg3 - 7.0*V2_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg3 - 4.0*V2_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg2 - 7.0*V2_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2*tdOmg3 - 7.0*V2_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg3 - 70.0*V2_dt*du_Omg1_dt*tdOmg1*tdOmg3 - 168.0*V2_dt*du_Omg1_dt*tdOmg2 - 7.0*V2_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2*tdOmg3 - 4.0*V2_dt*du_Omg2_dt*tdOmg1*pow2_tdOmg2 + 42.0*V2_dt*du_Omg2_dt*tdOmg1 - 7.0*V2_dt*du_Omg2_dt*pow3_tdOmg2*tdOmg3 - 7.0*V2_dt*du_Omg2_dt*tdOmg2*pow3_tdOmg3 - 98.0*V2_dt*du_Omg2_dt*tdOmg2*tdOmg3 - 7.0*V2_dt*du_Omg3_dt*pow2_tdOmg1*pow2_tdOmg3 + 7.0*V2_dt*du_Omg3_dt*pow2_tdOmg1 - 4.0*V2_dt*du_Omg3_dt*tdOmg1*tdOmg2*tdOmg3 - 7.0*V2_dt*du_Omg3_dt*pow2_tdOmg2*pow2_tdOmg3 + 35.0*V2_dt*du_Omg3_dt*pow2_tdOmg2 - 7.0*V2_dt*du_Omg3_dt*pow4_tdOmg3 - 63.0*V2_dt*du_Omg3_dt*pow2_tdOmg3 + 7.0*V3_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg2 - 4.0*V3_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg3 + 7.0*V3_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg2 + 7.0*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2*pow2_tdOmg3 + 70.0*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2 - 168.0*V3_dt*du_Omg1_dt*tdOmg3 + 7.0*V3_dt*du_Omg2_dt*pow2_tdOmg1*pow2_tdOmg2 - 7.0*V3_dt*du_Omg2_dt*pow2_tdOmg1 - 4.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2*tdOmg3 + 7.0*V3_dt*du_Omg2_dt*pow4_tdOmg2 + 7.0*V3_dt*du_Omg2_dt*pow2_tdOmg2*pow2_tdOmg3 + 63.0*V3_dt*du_Omg2_dt*pow2_tdOmg2 - 35.0*V3_dt*du_Omg2_dt*pow2_tdOmg3 + 7.0*V3_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg2*tdOmg3 - 4.0*V3_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg3 + 42.0*V3_dt*du_Omg3_dt*tdOmg1 + 7.0*V3_dt*du_Omg3_dt*pow3_tdOmg2*tdOmg3 + 7.0*V3_dt*du_Omg3_dt*tdOmg2*pow3_tdOmg3 + 98.0*V3_dt*du_Omg3_dt*tdOmg2*tdOmg3)
    out[1] = -0.013888888888888888*V1_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg3 + 0.016666666666666666*V1_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg2 - 0.013888888888888888*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2*tdOmg3 - 0.013888888888888888*V1_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg3 - 0.083333333333333329*V1_dt*du_Omg1_dt*tdOmg1*tdOmg3 - 0.16666666666666666*V1_dt*du_Omg1_dt*tdOmg2 - 0.013888888888888888*V1_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2*tdOmg3 + 0.016666666666666666*V1_dt*du_Omg2_dt*tdOmg1*pow2_tdOmg2 + 0.33333333333333331*V1_dt*du_Omg2_dt*tdOmg1 - 0.013888888888888888*V1_dt*du_Omg2_dt*pow3_tdOmg2*tdOmg3 - 0.013888888888888888*V1_dt*du_Omg2_dt*tdOmg2*pow3_tdOmg3 - 0.013888888888888888*V1_dt*du_Omg3_dt*pow2_tdOmg1*pow2_tdOmg3 + 0.125*V1_dt*du_Omg3_dt*pow2_tdOmg1 + 0.016666666666666666*V1_dt*du_Omg3_dt*tdOmg1*tdOmg2*tdOmg3 - 0.013888888888888888*V1_dt*du_Omg3_dt*pow2_tdOmg2*pow2_tdOmg3 + 0.041666666666666664*V1_dt*du_Omg3_dt*pow2_tdOmg2 - 0.013888888888888888*V1_dt*du_Omg3_dt*pow4_tdOmg3 + 0.041666666666666664*V1_dt*du_Omg3_dt*pow2_tdOmg3 - 0.5*V1_dt*du_Omg3_dt - 0.016666666666666666*V2_dt*du_Omg1_dt*pow3_tdOmg1 - 0.016666666666666666*V2_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg3 - 0.16666666666666666*V2_dt*du_Omg1_dt*tdOmg1 - 0.083333333333333329*V2_dt*du_Omg1_dt*tdOmg2*tdOmg3 - 0.016666666666666666*V2_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2 - 0.016666666666666666*V2_dt*du_Omg2_dt*tdOmg2*pow2_tdOmg3 - 0.016666666666666666*V2_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg3 + 0.083333333333333329*V2_dt*du_Omg3_dt*tdOmg1*tdOmg2 - 0.016666666666666666*V2_dt*du_Omg3_dt*pow3_tdOmg3 - 0.16666666666666666*V2_dt*du_Omg3_dt*tdOmg3 + 0.013888888888888888*V3_dt*du_Omg1_dt*pow4_tdOmg1 + 0.013888888888888888*V3_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg2 + 0.013888888888888888*V3_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg3 - 0.041666666666666664*V3_dt*du_Omg1_dt*pow2_tdOmg1 + 0.016666666666666666*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2*tdOmg3 - 0.041666666666666664*V3_dt*du_Omg1_dt*pow2_tdOmg2 - 0.125*V3_dt*du_Omg1_dt*pow2_tdOmg3 + 0.5*V3_dt*du_Omg1_dt + 0.013888888888888888*V3_dt*du_Omg2_dt*pow3_tdOmg1*tdOmg2 + 0.013888888888888888*V3_dt*du_Omg2_dt*tdOmg1*pow3_tdOmg2 + 0.013888888888888888*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2*pow2_tdOmg3 + 0.016666666666666666*V3_dt*du_Omg2_dt*pow2_tdOmg2*tdOmg3 + 0.33333333333333331*V3_dt*du_Omg2_dt*tdOmg3 + 0.013888888888888888*V3_dt*du_Omg3_dt*pow3_tdOmg1*tdOmg3 + 0.013888888888888888*V3_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg2*tdOmg3 + 0.013888888888888888*V3_dt*du_Omg3_dt*tdOmg1*pow3_tdOmg3 + 0.083333333333333329*V3_dt*du_Omg3_dt*tdOmg1*tdOmg3 + 0.016666666666666666*V3_dt*du_Omg3_dt*tdOmg2*pow2_tdOmg3 - 0.16666666666666666*V3_dt*du_Omg3_dt*tdOmg2 + 8.3507027951472401e-9*pow6_psi*(253.0*V1_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg3 - 24.0*V1_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg2 + 253.0*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2*tdOmg3 + 253.0*V1_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg3 + 8382.0*V1_dt*du_Omg1_dt*tdOmg1*tdOmg3 + 330.0*V1_dt*du_Omg1_dt*tdOmg2 + 253.0*V1_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2*tdOmg3 - 24.0*V1_dt*du_Omg2_dt*tdOmg1*pow2_tdOmg2 - 2640.0*V1_dt*du_Omg2_dt*tdOmg1 + 253.0*V1_dt*du_Omg2_dt*pow3_tdOmg2*tdOmg3 + 253.0*V1_dt*du_Omg2_dt*tdOmg2*pow3_tdOmg3 + 8118.0*V1_dt*du_Omg2_dt*tdOmg2*tdOmg3 + 253.0*V1_dt*du_Omg3_dt*pow2_tdOmg1*pow2_tdOmg3 - 297.0*V1_dt*du_Omg3_dt*pow2_tdOmg1 - 24.0*V1_dt*du_Omg3_dt*tdOmg1*tdOmg2*tdOmg3 + 253.0*V1_dt*du_Omg3_dt*pow2_tdOmg2*pow2_tdOmg3 - 33.0*V1_dt*du_Omg3_dt*pow2_tdOmg2 + 253.0*V1_dt*du_Omg3_dt*pow4_tdOmg3 + 8085.0*V1_dt*du_Omg3_dt*pow2_tdOmg3 + 24.0*V2_dt*du_Omg1_dt*pow3_tdOmg1 + 24.0*V2_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg3 + 2310.0*V2_dt*du_Omg1_dt*tdOmg1 + 264.0*V2_dt*du_Omg1_dt*tdOmg2*tdOmg3 + 24.0*V2_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2 + 24.0*V2_dt*du_Omg2_dt*tdOmg2*pow2_tdOmg3 + 24.0*V2_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg3 - 264.0*V2_dt*du_Omg3_dt*tdOmg1*tdOmg2 + 24.0*V2_dt*du_Omg3_dt*pow3_tdOmg3 + 2310.0*V2_dt*du_Omg3_dt*tdOmg3 - 253.0*V3_dt*du_Omg1_dt*pow4_tdOmg1 - 253.0*V3_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg2 - 253.0*V3_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg3 - 8085.0*V3_dt*du_Omg1_dt*pow2_tdOmg1 - 24.0*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2*tdOmg3 + 33.0*V3_dt*du_Omg1_dt*pow2_tdOmg2 + 297.0*V3_dt*du_Omg1_dt*pow2_tdOmg3 - 253.0*V3_dt*du_Omg2_dt*pow3_tdOmg1*tdOmg2 - 253.0*V3_dt*du_Omg2_dt*tdOmg1*pow3_tdOmg2 - 253.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2*pow2_tdOmg3 - 8118.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2 - 24.0*V3_dt*du_Omg2_dt*pow2_tdOmg2*tdOmg3 - 2640.0*V3_dt*du_Omg2_dt*tdOmg3 - 253.0*V3_dt*du_Omg3_dt*pow3_tdOmg1*tdOmg3 - 253.0*V3_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg2*tdOmg3 - 253.0*V3_dt*du_Omg3_dt*tdOmg1*pow3_tdOmg3 - 8382.0*V3_dt*du_Omg3_dt*tdOmg1*tdOmg3 - 24.0*V3_dt*du_Omg3_dt*tdOmg2*pow2_tdOmg3 + 330.0*V3_dt*du_Omg3_dt*tdOmg2) + 1.6534391534391535e-6*pow4_psi*(-41.0*V1_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg3 + 10.0*V1_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg2 - 41.0*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2*tdOmg3 - 41.0*V1_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg3 - 930.0*V1_dt*du_Omg1_dt*tdOmg1*tdOmg3 - 120.0*V1_dt*du_Omg1_dt*tdOmg2 - 41.0*V1_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2*tdOmg3 + 10.0*V1_dt*du_Omg2_dt*tdOmg1*pow2_tdOmg2 + 720.0*V1_dt*du_Omg2_dt*tdOmg1 - 41.0*V1_dt*du_Omg2_dt*pow3_tdOmg2*tdOmg3 - 41.0*V1_dt*du_Omg2_dt*tdOmg2*pow3_tdOmg3 - 840.0*V1_dt*du_Omg2_dt*tdOmg2*tdOmg3 - 41.0*V1_dt*du_Omg3_dt*pow2_tdOmg1*pow2_tdOmg3 + 105.0*V1_dt*du_Omg3_dt*pow2_tdOmg1 + 10.0*V1_dt*du_Omg3_dt*tdOmg1*tdOmg2*tdOmg3 - 41.0*V1_dt*du_Omg3_dt*pow2_tdOmg2*pow2_tdOmg3 + 15.0*V1_dt*du_Omg3_dt*pow2_tdOmg2 - 41.0*V1_dt*du_Omg3_dt*pow4_tdOmg3 - 825.0*V1_dt*du_Omg3_dt*pow2_tdOmg3 - 10.0*V2_dt*du_Omg1_dt*pow3_tdOmg1 - 10.0*V2_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg3 - 600.0*V2_dt*du_Omg1_dt*tdOmg1 - 90.0*V2_dt*du_Omg1_dt*tdOmg2*tdOmg3 - 10.0*V2_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2 - 10.0*V2_dt*du_Omg2_dt*tdOmg2*pow2_tdOmg3 - 10.0*V2_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg3 + 90.0*V2_dt*du_Omg3_dt*tdOmg1*tdOmg2 - 10.0*V2_dt*du_Omg3_dt*pow3_tdOmg3 - 600.0*V2_dt*du_Omg3_dt*tdOmg3 + 41.0*V3_dt*du_Omg1_dt*pow4_tdOmg1 + 41.0*V3_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg2 + 41.0*V3_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg3 + 825.0*V3_dt*du_Omg1_dt*pow2_tdOmg1 + 10.0*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2*tdOmg3 - 15.0*V3_dt*du_Omg1_dt*pow2_tdOmg2 - 105.0*V3_dt*du_Omg1_dt*pow2_tdOmg3 + 41.0*V3_dt*du_Omg2_dt*pow3_tdOmg1*tdOmg2 + 41.0*V3_dt*du_Omg2_dt*tdOmg1*pow3_tdOmg2 + 41.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2*pow2_tdOmg3 + 840.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2 + 10.0*V3_dt*du_Omg2_dt*pow2_tdOmg2*tdOmg3 + 720.0*V3_dt*du_Omg2_dt*tdOmg3 + 41.0*V3_dt*du_Omg3_dt*pow3_tdOmg1*tdOmg3 + 41.0*V3_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg2*tdOmg3 + 41.0*V3_dt*du_Omg3_dt*tdOmg1*pow3_tdOmg3 + 930.0*V3_dt*du_Omg3_dt*tdOmg1*tdOmg3 + 10.0*V3_dt*du_Omg3_dt*tdOmg2*pow2_tdOmg3 - 120.0*V3_dt*du_Omg3_dt*tdOmg2) + 0.00019841269841269841*pow2_psi*(7.0*V1_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg3 - 4.0*V1_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg2 + 7.0*V1_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2*tdOmg3 + 7.0*V1_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg3 + 98.0*V1_dt*du_Omg1_dt*tdOmg1*tdOmg3 + 42.0*V1_dt*du_Omg1_dt*tdOmg2 + 7.0*V1_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2*tdOmg3 - 4.0*V1_dt*du_Omg2_dt*tdOmg1*pow2_tdOmg2 - 168.0*V1_dt*du_Omg2_dt*tdOmg1 + 7.0*V1_dt*du_Omg2_dt*pow3_tdOmg2*tdOmg3 + 7.0*V1_dt*du_Omg2_dt*tdOmg2*pow3_tdOmg3 + 70.0*V1_dt*du_Omg2_dt*tdOmg2*tdOmg3 + 7.0*V1_dt*du_Omg3_dt*pow2_tdOmg1*pow2_tdOmg3 - 35.0*V1_dt*du_Omg3_dt*pow2_tdOmg1 - 4.0*V1_dt*du_Omg3_dt*tdOmg1*tdOmg2*tdOmg3 + 7.0*V1_dt*du_Omg3_dt*pow2_tdOmg2*pow2_tdOmg3 - 7.0*V1_dt*du_Omg3_dt*pow2_tdOmg2 + 7.0*V1_dt*du_Omg3_dt*pow4_tdOmg3 + 63.0*V1_dt*du_Omg3_dt*pow2_tdOmg3 + 4.0*V2_dt*du_Omg1_dt*pow3_tdOmg1 + 4.0*V2_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg3 + 126.0*V2_dt*du_Omg1_dt*tdOmg1 + 28.0*V2_dt*du_Omg1_dt*tdOmg2*tdOmg3 + 4.0*V2_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2 + 4.0*V2_dt*du_Omg2_dt*tdOmg2*pow2_tdOmg3 + 4.0*V2_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg3 - 28.0*V2_dt*du_Omg3_dt*tdOmg1*tdOmg2 + 4.0*V2_dt*du_Omg3_dt*pow3_tdOmg3 + 126.0*V2_dt*du_Omg3_dt*tdOmg3 - 7.0*V3_dt*du_Omg1_dt*pow4_tdOmg1 - 7.0*V3_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg2 - 7.0*V3_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg3 - 63.0*V3_dt*du_Omg1_dt*pow2_tdOmg1 - 4.0*V3_dt*du_Omg1_dt*tdOmg1*tdOmg2*tdOmg3 + 7.0*V3_dt*du_Omg1_dt*pow2_tdOmg2 + 35.0*V3_dt*du_Omg1_dt*pow2_tdOmg3 - 7.0*V3_dt*du_Omg2_dt*pow3_tdOmg1*tdOmg2 - 7.0*V3_dt*du_Omg2_dt*tdOmg1*pow3_tdOmg2 - 7.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2*pow2_tdOmg3 - 70.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg2 - 4.0*V3_dt*du_Omg2_dt*pow2_tdOmg2*tdOmg3 - 168.0*V3_dt*du_Omg2_dt*tdOmg3 - 7.0*V3_dt*du_Omg3_dt*pow3_tdOmg1*tdOmg3 - 7.0*V3_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg2*tdOmg3 - 7.0*V3_dt*du_Omg3_dt*tdOmg1*pow3_tdOmg3 - 98.0*V3_dt*du_Omg3_dt*tdOmg1*tdOmg3 - 4.0*V3_dt*du_Omg3_dt*tdOmg2*pow2_tdOmg3 + 42.0*V3_dt*du_Omg3_dt*tdOmg2)
    out[2] = 0.013888888888888888*V1_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg2 + 0.016666666666666666*V1_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg3 + 0.013888888888888888*V1_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg2 + 0.013888888888888888*V1_dt*du_Omg1_dt*tdOmg1*tdOmg2*pow2_tdOmg3 + 0.083333333333333329*V1_dt*du_Omg1_dt*tdOmg1*tdOmg2 - 0.16666666666666666*V1_dt*du_Omg1_dt*tdOmg3 + 0.013888888888888888*V1_dt*du_Omg2_dt*pow2_tdOmg1*pow2_tdOmg2 - 0.125*V1_dt*du_Omg2_dt*pow2_tdOmg1 + 0.016666666666666666*V1_dt*du_Omg2_dt*tdOmg1*tdOmg2*tdOmg3 + 0.013888888888888888*V1_dt*du_Omg2_dt*pow4_tdOmg2 + 0.013888888888888888*V1_dt*du_Omg2_dt*pow2_tdOmg2*pow2_tdOmg3 - 0.041666666666666664*V1_dt*du_Omg2_dt*pow2_tdOmg2 - 0.041666666666666664*V1_dt*du_Omg2_dt*pow2_tdOmg3 + 0.5*V1_dt*du_Omg2_dt + 0.013888888888888888*V1_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg2*tdOmg3 + 0.016666666666666666*V1_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg3 + 0.33333333333333331*V1_dt*du_Omg3_dt*tdOmg1 + 0.013888888888888888*V1_dt*du_Omg3_dt*pow3_tdOmg2*tdOmg3 + 0.013888888888888888*V1_dt*du_Omg3_dt*tdOmg2*pow3_tdOmg3 - 0.013888888888888888*V2_dt*du_Omg1_dt*pow4_tdOmg1 - 0.013888888888888888*V2_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg2 - 0.013888888888888888*V2_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg3 + 0.041666666666666664*V2_dt*du_Omg1_dt*pow2_tdOmg1 + 0.016666666666666666*V2_dt*du_Omg1_dt*tdOmg1*tdOmg2*tdOmg3 + 0.125*V2_dt*du_Omg1_dt*pow2_tdOmg2 + 0.041666666666666664*V2_dt*du_Omg1_dt*pow2_tdOmg3 - 0.5*V2_dt*du_Omg1_dt - 0.013888888888888888*V2_dt*du_Omg2_dt*pow3_tdOmg1*tdOmg2 - 0.013888888888888888*V2_dt*du_Omg2_dt*tdOmg1*pow3_tdOmg2 - 0.013888888888888888*V2_dt*du_Omg2_dt*tdOmg1*tdOmg2*pow2_tdOmg3 - 0.083333333333333329*V2_dt*du_Omg2_dt*tdOmg1*tdOmg2 + 0.016666666666666666*V2_dt*du_Omg2_dt*pow2_tdOmg2*tdOmg3 - 0.16666666666666666*V2_dt*du_Omg2_dt*tdOmg3 - 0.013888888888888888*V2_dt*du_Omg3_dt*pow3_tdOmg1*tdOmg3 - 0.013888888888888888*V2_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg2*tdOmg3 - 0.013888888888888888*V2_dt*du_Omg3_dt*tdOmg1*pow3_tdOmg3 + 0.016666666666666666*V2_dt*du_Omg3_dt*tdOmg2*pow2_tdOmg3 + 0.33333333333333331*V2_dt*du_Omg3_dt*tdOmg2 - 0.016666666666666666*V3_dt*du_Omg1_dt*pow3_tdOmg1 - 0.016666666666666666*V3_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2 - 0.16666666666666666*V3_dt*du_Omg1_dt*tdOmg1 + 0.083333333333333329*V3_dt*du_Omg1_dt*tdOmg2*tdOmg3 - 0.016666666666666666*V3_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2 - 0.083333333333333329*V3_dt*du_Omg2_dt*tdOmg1*tdOmg3 - 0.016666666666666666*V3_dt*du_Omg2_dt*pow3_tdOmg2 - 0.16666666666666666*V3_dt*du_Omg2_dt*tdOmg2 - 0.016666666666666666*V3_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg3 - 0.016666666666666666*V3_dt*du_Omg3_dt*pow2_tdOmg2*tdOmg3 + 8.3507027951472401e-9*pow6_psi*(-253.0*V1_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg2 - 24.0*V1_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg3 - 253.0*V1_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg2 - 253.0*V1_dt*du_Omg1_dt*tdOmg1*tdOmg2*pow2_tdOmg3 - 8382.0*V1_dt*du_Omg1_dt*tdOmg1*tdOmg2 + 330.0*V1_dt*du_Omg1_dt*tdOmg3 - 253.0*V1_dt*du_Omg2_dt*pow2_tdOmg1*pow2_tdOmg2 + 297.0*V1_dt*du_Omg2_dt*pow2_tdOmg1 - 24.0*V1_dt*du_Omg2_dt*tdOmg1*tdOmg2*tdOmg3 - 253.0*V1_dt*du_Omg2_dt*pow4_tdOmg2 - 253.0*V1_dt*du_Omg2_dt*pow2_tdOmg2*pow2_tdOmg3 - 8085.0*V1_dt*du_Omg2_dt*pow2_tdOmg2 + 33.0*V1_dt*du_Omg2_dt*pow2_tdOmg3 - 253.0*V1_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg2*tdOmg3 - 24.0*V1_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg3 - 2640.0*V1_dt*du_Omg3_dt*tdOmg1 - 253.0*V1_dt*du_Omg3_dt*pow3_tdOmg2*tdOmg3 - 253.0*V1_dt*du_Omg3_dt*tdOmg2*pow3_tdOmg3 - 8118.0*V1_dt*du_Omg3_dt*tdOmg2*tdOmg3 + 253.0*V2_dt*du_Omg1_dt*pow4_tdOmg1 + 253.0*V2_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg2 + 253.0*V2_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg3 + 8085.0*V2_dt*du_Omg1_dt*pow2_tdOmg1 - 24.0*V2_dt*du_Omg1_dt*tdOmg1*tdOmg2*tdOmg3 - 297.0*V2_dt*du_Omg1_dt*pow2_tdOmg2 - 33.0*V2_dt*du_Omg1_dt*pow2_tdOmg3 + 253.0*V2_dt*du_Omg2_dt*pow3_tdOmg1*tdOmg2 + 253.0*V2_dt*du_Omg2_dt*tdOmg1*pow3_tdOmg2 + 253.0*V2_dt*du_Omg2_dt*tdOmg1*tdOmg2*pow2_tdOmg3 + 8382.0*V2_dt*du_Omg2_dt*tdOmg1*tdOmg2 - 24.0*V2_dt*du_Omg2_dt*pow2_tdOmg2*tdOmg3 + 330.0*V2_dt*du_Omg2_dt*tdOmg3 + 253.0*V2_dt*du_Omg3_dt*pow3_tdOmg1*tdOmg3 + 253.0*V2_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg2*tdOmg3 + 253.0*V2_dt*du_Omg3_dt*tdOmg1*pow3_tdOmg3 + 8118.0*V2_dt*du_Omg3_dt*tdOmg1*tdOmg3 - 24.0*V2_dt*du_Omg3_dt*tdOmg2*pow2_tdOmg3 - 2640.0*V2_dt*du_Omg3_dt*tdOmg2 + 24.0*V3_dt*du_Omg1_dt*pow3_tdOmg1 + 24.0*V3_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2 + 2310.0*V3_dt*du_Omg1_dt*tdOmg1 - 264.0*V3_dt*du_Omg1_dt*tdOmg2*tdOmg3 + 24.0*V3_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2 + 264.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg3 + 24.0*V3_dt*du_Omg2_dt*pow3_tdOmg2 + 2310.0*V3_dt*du_Omg2_dt*tdOmg2 + 24.0*V3_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg3 + 24.0*V3_dt*du_Omg3_dt*pow2_tdOmg2*tdOmg3) + 1.6534391534391535e-6*pow4_psi*(41.0*V1_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg2 + 10.0*V1_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg3 + 41.0*V1_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg2 + 41.0*V1_dt*du_Omg1_dt*tdOmg1*tdOmg2*pow2_tdOmg3 + 930.0*V1_dt*du_Omg1_dt*tdOmg1*tdOmg2 - 120.0*V1_dt*du_Omg1_dt*tdOmg3 + 41.0*V1_dt*du_Omg2_dt*pow2_tdOmg1*pow2_tdOmg2 - 105.0*V1_dt*du_Omg2_dt*pow2_tdOmg1 + 10.0*V1_dt*du_Omg2_dt*tdOmg1*tdOmg2*tdOmg3 + 41.0*V1_dt*du_Omg2_dt*pow4_tdOmg2 + 41.0*V1_dt*du_Omg2_dt*pow2_tdOmg2*pow2_tdOmg3 + 825.0*V1_dt*du_Omg2_dt*pow2_tdOmg2 - 15.0*V1_dt*du_Omg2_dt*pow2_tdOmg3 + 41.0*V1_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg2*tdOmg3 + 10.0*V1_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg3 + 720.0*V1_dt*du_Omg3_dt*tdOmg1 + 41.0*V1_dt*du_Omg3_dt*pow3_tdOmg2*tdOmg3 + 41.0*V1_dt*du_Omg3_dt*tdOmg2*pow3_tdOmg3 + 840.0*V1_dt*du_Omg3_dt*tdOmg2*tdOmg3 - 41.0*V2_dt*du_Omg1_dt*pow4_tdOmg1 - 41.0*V2_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg2 - 41.0*V2_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg3 - 825.0*V2_dt*du_Omg1_dt*pow2_tdOmg1 + 10.0*V2_dt*du_Omg1_dt*tdOmg1*tdOmg2*tdOmg3 + 105.0*V2_dt*du_Omg1_dt*pow2_tdOmg2 + 15.0*V2_dt*du_Omg1_dt*pow2_tdOmg3 - 41.0*V2_dt*du_Omg2_dt*pow3_tdOmg1*tdOmg2 - 41.0*V2_dt*du_Omg2_dt*tdOmg1*pow3_tdOmg2 - 41.0*V2_dt*du_Omg2_dt*tdOmg1*tdOmg2*pow2_tdOmg3 - 930.0*V2_dt*du_Omg2_dt*tdOmg1*tdOmg2 + 10.0*V2_dt*du_Omg2_dt*pow2_tdOmg2*tdOmg3 - 120.0*V2_dt*du_Omg2_dt*tdOmg3 - 41.0*V2_dt*du_Omg3_dt*pow3_tdOmg1*tdOmg3 - 41.0*V2_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg2*tdOmg3 - 41.0*V2_dt*du_Omg3_dt*tdOmg1*pow3_tdOmg3 - 840.0*V2_dt*du_Omg3_dt*tdOmg1*tdOmg3 + 10.0*V2_dt*du_Omg3_dt*tdOmg2*pow2_tdOmg3 + 720.0*V2_dt*du_Omg3_dt*tdOmg2 - 10.0*V3_dt*du_Omg1_dt*pow3_tdOmg1 - 10.0*V3_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2 - 600.0*V3_dt*du_Omg1_dt*tdOmg1 + 90.0*V3_dt*du_Omg1_dt*tdOmg2*tdOmg3 - 10.0*V3_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2 - 90.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg3 - 10.0*V3_dt*du_Omg2_dt*pow3_tdOmg2 - 600.0*V3_dt*du_Omg2_dt*tdOmg2 - 10.0*V3_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg3 - 10.0*V3_dt*du_Omg3_dt*pow2_tdOmg2*tdOmg3) + 0.00019841269841269841*pow2_psi*(-7.0*V1_dt*du_Omg1_dt*pow3_tdOmg1*tdOmg2 - 4.0*V1_dt*du_Omg1_dt*pow2_tdOmg1*tdOmg3 - 7.0*V1_dt*du_Omg1_dt*tdOmg1*pow3_tdOmg2 - 7.0*V1_dt*du_Omg1_dt*tdOmg1*tdOmg2*pow2_tdOmg3 - 98.0*V1_dt*du_Omg1_dt*tdOmg1*tdOmg2 + 42.0*V1_dt*du_Omg1_dt*tdOmg3 - 7.0*V1_dt*du_Omg2_dt*pow2_tdOmg1*pow2_tdOmg2 + 35.0*V1_dt*du_Omg2_dt*pow2_tdOmg1 - 4.0*V1_dt*du_Omg2_dt*tdOmg1*tdOmg2*tdOmg3 - 7.0*V1_dt*du_Omg2_dt*pow4_tdOmg2 - 7.0*V1_dt*du_Omg2_dt*pow2_tdOmg2*pow2_tdOmg3 - 63.0*V1_dt*du_Omg2_dt*pow2_tdOmg2 + 7.0*V1_dt*du_Omg2_dt*pow2_tdOmg3 - 7.0*V1_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg2*tdOmg3 - 4.0*V1_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg3 - 168.0*V1_dt*du_Omg3_dt*tdOmg1 - 7.0*V1_dt*du_Omg3_dt*pow3_tdOmg2*tdOmg3 - 7.0*V1_dt*du_Omg3_dt*tdOmg2*pow3_tdOmg3 - 70.0*V1_dt*du_Omg3_dt*tdOmg2*tdOmg3 + 7.0*V2_dt*du_Omg1_dt*pow4_tdOmg1 + 7.0*V2_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg2 + 7.0*V2_dt*du_Omg1_dt*pow2_tdOmg1*pow2_tdOmg3 + 63.0*V2_dt*du_Omg1_dt*pow2_tdOmg1 - 4.0*V2_dt*du_Omg1_dt*tdOmg1*tdOmg2*tdOmg3 - 35.0*V2_dt*du_Omg1_dt*pow2_tdOmg2 - 7.0*V2_dt*du_Omg1_dt*pow2_tdOmg3 + 7.0*V2_dt*du_Omg2_dt*pow3_tdOmg1*tdOmg2 + 7.0*V2_dt*du_Omg2_dt*tdOmg1*pow3_tdOmg2 + 7.0*V2_dt*du_Omg2_dt*tdOmg1*tdOmg2*pow2_tdOmg3 + 98.0*V2_dt*du_Omg2_dt*tdOmg1*tdOmg2 - 4.0*V2_dt*du_Omg2_dt*pow2_tdOmg2*tdOmg3 + 42.0*V2_dt*du_Omg2_dt*tdOmg3 + 7.0*V2_dt*du_Omg3_dt*pow3_tdOmg1*tdOmg3 + 7.0*V2_dt*du_Omg3_dt*tdOmg1*pow2_tdOmg2*tdOmg3 + 7.0*V2_dt*du_Omg3_dt*tdOmg1*pow3_tdOmg3 + 70.0*V2_dt*du_Omg3_dt*tdOmg1*tdOmg3 - 4.0*V2_dt*du_Omg3_dt*tdOmg2*pow2_tdOmg3 - 168.0*V2_dt*du_Omg3_dt*tdOmg2 + 4.0*V3_dt*du_Omg1_dt*pow3_tdOmg1 + 4.0*V3_dt*du_Omg1_dt*tdOmg1*pow2_tdOmg2 + 126.0*V3_dt*du_Omg1_dt*tdOmg1 - 28.0*V3_dt*du_Omg1_dt*tdOmg2*tdOmg3 + 4.0*V3_dt*du_Omg2_dt*pow2_tdOmg1*tdOmg2 + 28.0*V3_dt*du_Omg2_dt*tdOmg1*tdOmg3 + 4.0*V3_dt*du_Omg2_dt*pow3_tdOmg2 + 126.0*V3_dt*du_Omg2_dt*tdOmg2 + 4.0*V3_dt*du_Omg3_dt*pow2_tdOmg1*tdOmg3 + 4.0*V3_dt*du_Omg3_dt*pow2_tdOmg2*tdOmg3)
    
    clear_buffer(psi.shape, 5)
    clear_buffer(Omg.shape, 6)

    return out

def reconstruct_frame(th, pi_hat, L0, R_u0=None, E_u0=None, out=None, dus=None):
    if len(pi_hat.shape) != 3:
        raise Exception('pi_hat has wrong shape.')

    Mm = th.shape[-1]

    if dus is None:
        dus = np.zeros(Mm-1, dtype=th.dtype)
        dus[:] = get_du(Mm, L0)

    if out is None:
        out = np.zeros((4, 4, Mm), dtype=th.dtype)
        out[0,0,0] = 1

    if not R_u0 is None:
        out[1:,0,0] = R_u0
    else:
        out[1:,0,0] = 0

    if not E_u0 is None:
        out[1:,1:,0] = E_u0
    else:
        out[1:,1:,0] = np.eye(3)

    X, exp_X = get_buffers((4,4,Mm-1), 2)

    X = construct_se3_elem(th[...,:-1], pi_hat[...,:-1], out=X)

    exp_X = compute_exp_se3_variable_d(X, dus, out=exp_X)

    for i in range(1, Mm):
        out[:,:,i] = out[:,:,i-1].dot(exp_X[:,:,i-1])

    E = out[1:,1:]
    R = out[1:,0]

    clear_buffer((4,4,Mm-1), 2)

    return out, R, E

def compute_exp_se3_variable_d(X, d, taylor_tol=1e-2, out=None, w_norm=None, psi=None, taylor_mask=None):
    w = X[1:,1:]

    if w_norm is None:
        w_norm = so3_norm(w)

    if psi is None:
        psi = d*w_norm

    if taylor_mask is None:
        taylor_mask = psi < taylor_tol

    if len(X.shape) == 3:
        if out is None:
            out = np.zeros((4,4,w.shape[-1]), dtype=X.dtype)

        N_taylor = np.count_nonzero(taylor_mask)
        taylor_ratio = N_taylor/w.shape[-1]

        N_zeroes = np.count_nonzero(psi == 0)

        if N_zeroes > 0:
            if taylor_ratio != 1:
                out[...,~taylor_mask] = compute_exp_se3_analytic(X[...,~taylor_mask], d[~taylor_mask], w_norm=w_norm[...,~taylor_mask], psi=psi[...,~taylor_mask])
            out[...,taylor_mask] = compute_exp_se3_taylor(X[...,taylor_mask], d[taylor_mask], w_norm=w_norm[...,taylor_mask], psi=psi[...,taylor_mask])
        else:
            if taylor_ratio > 0.5:
                compute_exp_se3_taylor(X, d, out=out, w_norm=w_norm, psi=psi)
                if taylor_ratio != 1:
                    out[...,~taylor_mask] = compute_exp_se3_analytic(X[...,~taylor_mask], d[~taylor_mask], w_norm=w_norm[...,~taylor_mask], psi=psi[...,~taylor_mask])
            else:
                compute_exp_se3_analytic(X, d, out=out, w_norm=w_norm, psi=psi)
                if taylor_ratio != 0:
                    out[...,taylor_mask] = compute_exp_se3_taylor(X[...,taylor_mask], d[taylor_mask], w_norm=w_norm[...,taylor_mask], psi=psi[...,taylor_mask])
    else:
        if not taylor_mask:
            out = compute_exp_se3_analytic(X, d)
        else:
            out = compute_exp_se3_taylor(X, d)

    return out

def integrate_frame_forward(Fr, V, Omg, dt, Omg_is_matrix=False):
    if Omg_is_matrix:
        Omg_hat = Omg
    else:
        Omg_hat = hat_vec_to_mat(Omg)
    Y = construct_se3_elem(V, Omg_hat)
    exp_Y = compute_exp_se3(Y, dt)
    Fr = Fr.dot(exp_Y)
    return Fr

def OLD_compute_closure_failure(th, pi, L0):
    Nm = pi.shape[-1]
    pi_hat = hat_vec_to_mat(pi)
    F, R, E = reconstruct_frame(th, pi_hat, L0)
    du = get_du(Nm, L0)

    th_uf = th[...,-1]
    pi_hat_uf = pi_hat[...,-1]
    R_uf = R[...,-1]
    E_uf = E[...,-1]
    F_uf = construct_oriented_frame(R_uf, E_uf)

    F_u0_proj = integrate_frame_forward(F_uf, th_uf, pi_hat_uf, du, Omg_is_matrix=True)
    R_u0_proj, E_u0_proj =  get_R_and_frame(F_u0_proj)

    close_err = np.sum(R_u0_proj**2 + (E_u0_proj-np.eye(3))**2)
    return close_err, R_u0_proj, E_u0_proj

def compute_closure_failure(th, pi, R, E, L0):
    Nm = pi.shape[-1]
    du = get_du(Nm, L0)

    th_uf = th[...,-1]
    pi_uf = pi[...,-1]
    R_uf = R[...,-1]
    E_uf = E[...,-1]
    F_uf = construct_oriented_frame(R_uf, E_uf)

    F_u0_proj = integrate_frame_forward(F_uf, th_uf, pi_uf, du)
    R_u0_proj, E_u0_proj =  get_R_and_frame(F_u0_proj)

    R_u0 = R[...,0]
    E_u0 = E[...,0]

    close_err = np.sum((R_u0_proj-R_u0)**2 + (E_u0_proj-E_u0)**2)
    return close_err, R_u0_proj, E_u0_proj

def reconstruct_curve(c_th, c_pi, Fr_u0, Mm, Mm_save, path_handler_render, pre_transform=None):
    Mm_render = path_handler_render.Mm
    L0 = path_handler_render.L

    c_th_render = path_handler_render.change_Mm(c_th, Mm, Mm_render)
    c_pi_render = path_handler_render.change_Mm(c_pi, Mm, Mm_render)

    th_render = path_handler_render.iDT(c_th_render)
    pi_render = path_handler_render.iDT(c_pi_render)

    if not pre_transform is None:
        th_render, pi_render = pre_transform(th_render, pi_render)
    
    R_u0, E_u0 = get_R_and_frame(Fr_u0)
    Fr, R, E = reconstruct_frame(th_render, hat_vec_to_mat(pi_render), L0, R_u0, E_u0)

    Mm_save_step = int(Mm_render / Mm_save)
    if Mm_save_step != Mm_render / Mm_save:
        raise Exception('Mm_render and Mm_save incompatible')

    R_save = R[...,::Mm_save_step]
    E_save = E[...,::Mm_save_step]
    
    # Compute the extent to which the loop fails to close

    close_err, _, _ = compute_closure_failure(th_render, pi_render, R, E, L0)

    return R_save, E_save, close_err

def reconstruct_curve_cheb(th, pi, Fr_u0, Mm_render, Mm_save, path_handler, pre_transform=None):
    L0 = path_handler.L

    th_render = path_handler.cheb_to_unif(th, Mm_render)
    pi_render = path_handler.cheb_to_unif(pi, Mm_render)

    if not pre_transform is None:
        th_render, pi_render = pre_transform(th_render, pi_render)
    
    R_u0, E_u0 = get_R_and_frame(Fr_u0)
    Fr, R, E = reconstruct_frame(th_render, hat_vec_to_mat(pi_render), L0, R_u0, E_u0)

    Mm_save_step = int(Mm_render / Mm_save)
    if Mm_save_step != Mm_render / Mm_save:
        raise Exception('Mm_render and Mm_save incompatible')

    R_save = R[...,::Mm_save_step]
    E_save = E[...,::Mm_save_step]
    
    # Compute the extent to which the loop fails to close

    close_err, _, _ = compute_closure_failure(th_render, pi_render, R, E, L0)

    return R_save, E_save, close_err