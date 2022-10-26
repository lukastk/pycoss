import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

import pycoss.rod.interp.FFT as FFT
from pycoss.rod.helpers.geometry import *
from pycoss.rod.helpers.misc import *

def get_differential_invariants_from_curve(R, Nm, target_Mm, L0, L=None, twist=None, err_tol=1e-1, dR=None):
    dim = 3
    Mm = R.shape[-1]
    path_handler = FFT.FourierHandler(dim, Nm, Mm, L0)
    path_handler_1d = FFT.FourierHandler(1, Nm, Mm, L0)

    us = grid(Mm, L0)
    us_ext = np.linspace(0, L0, Mm+1, endpoint=True)
    du = us[1] - us[0]

    if dR is None:
        dR = path_handler.diff_f(R)

    if not L is None:
        h = norm(dR)
        h_ext = get_periodic_ext(h)
        int_h = np.trapz(h_ext, x=us_ext)

        R *= L/int_h
        dR *= L/int_h

    #### Find a compatible connection form

    h = norm(dR)
    h_ext = get_periodic_ext(h)
    int_h = np.trapz(h_ext, x=us_ext)

    t = dR/h
    dt = path_handler.diff_f(t)

    n = np.zeros(t.shape)
    n[0] = t[2]*t[0]
    n[1] = t[2]*t[1]
    n[2] = -t[0]**2 - t[1]**2
    _n_norm = norm(n)
    n /= _n_norm

    dn = path_handler.diff_f(n)

    b = np.zeros(t.shape)
    b[0] = n[2]*t[1] - n[1]*t[2]
    b[1] = -n[2]*t[0] + n[0]*t[2]
    b[2] = n[1]*t[0] - n[0]*t[1]

    db = path_handler.diff_f(b)

    pi = np.zeros((3, Mm))
    pi[0] = vec_vec_dot(dn,b)
    pi[1] = -vec_vec_dot(dt,b)
    pi[2] = vec_vec_dot(dt,n)

    pi1, pi2, pi3 = pi


    #### Remove the twist as much as possible

    phi = path_handler_1d.int_f(pi1)
    dphi = path_handler_1d.diff_f(phi) # Will in general not be equal to pi1

    _pi1, _pi2, _pi3 = pi

    pi3 = np.sin(phi)*_pi2 + np.cos(phi)*_pi3
    pi2 = np.cos(phi)*_pi2 - np.sin(phi)*_pi3
    pi1 = _pi1 - dphi


    #### Add specified twist

    if not twist is None:

        twist *= -1 # So that twist corresponds to pi1 

        dtwist = path_handler_1d.diff_f(twist)

        _pi1 = -dtwist
        _pi2 = pi2*np.cos(twist) - pi3*np.sin(twist)
        _pi3 = pi3*np.cos(twist) + pi2*np.sin(twist)

        pi1[:] = _pi1
        pi2[:] = _pi2
        pi3[:] = _pi3
        
        
    #### Construct spatial solder form and spatial connection form

    th = np.zeros((3, Mm))
    th[0] = h

    pi_hat = hat_vec_to_mat(pi)

    #### Resample the forms to only have Nm modes

    c_th = path_handler.DT(th)
    c_pi = path_handler.DT(pi)

    th = path_handler.iDT(c_th)
    pi = path_handler.iDT(c_pi)

    pi_hat = hat_vec_to_mat(pi)

    #### Reconstruct curve

    R_u0 = R[:,0]

    E_u0 = np.zeros((3,3))
    E_u0[:,0] = t[:,0]
    E_u0[:,1] = n[:,0]
    E_u0[:,2] = b[:,0]

    F, R_reconstructed, E = reconstruct_frame(th, pi_hat, L0, R_u0, E_u0)

    R_err = np.max(np.abs(R-R_reconstructed))

    # Complute the extent to which the loop fails to close

    th_uf = th[...,-1]
    pi_hat_uf = pi_hat[...,-1]
    R_uf = R_reconstructed[...,-1]
    E_uf = E[...,-1]
    F_uf = construct_oriented_frame(R_uf, E_uf)
    R_u0_proj, E_u0_proj =  get_R_and_frame(integrate_frame_forward(F_uf, th_uf, pi_hat_uf, du, Omg_is_matrix=True))

    close_err = np.linalg.norm(R_u0_proj - R_u0)

    # Reweight the coefficients to the target Mm

    target_Mm

    c_th = path_handler.change_Mm(c_th, Mm, target_Mm)
    c_pi = path_handler.change_Mm(c_pi, Mm, target_Mm)

    return c_th, c_pi, R_u0, E_u0, R_err, close_err
    
