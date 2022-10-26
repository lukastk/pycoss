from re import A
import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

import pycoss
from pycoss.rod.helpers.geometry import *

def KE_simulate(params):
    dim = 3

    path_handler = params['path_handler']

    Nm = path_handler.Nm
    Mm = path_handler.Mm
    L0 = path_handler.L

    T = params['T']
    dt = params['dt']

    integrator = params['integrator']

    c_pi = np.copy(params['c_pi0'])
    c_th = np.copy(params['c_th0'])

    lmbd = params['lmbd']

    if 'Fr0_u0' in params:
        Fr_u0 = np.copy(params['Fr0_u0'])
    else:
        Fr_u0 = construct_oriented_frame(np.zeros(dim), np.eye(dim))
    
    if 'N_save' in params:
        N_save = params['N_save']
        path_handler_render = params['path_handler_render']
        Mm_render = path_handler_render.Mm
        Mm_save = params['Mm_save']

        if 'pre_transform_render' in params:
            pre_transform_render = params['pre_transform_render']
        else:
            pre_transform_render = None
    else:
        N_save = -1

    save_differential_invariants = params['save_differential_invariants'] if 'save_differential_invariants' in params else False

    compute_V = params['V']
    compute_Omg = params['Omg']

    taylor_tol = params['taylor_tol']

    if 'N_print' in params: 
        N_print = params['N_print']
    else:
        N_print = 100

    if 'verbose' in params:
        verbose = params['verbose']
    else:
        verbose = True

    if 'pre_transform' in params:
        pre_transform = params['pre_transform']
    else:
        pre_transform = None

    if 'post_transform' in params:
        post_transform = params['post_transform']
    else:
        post_transform = None

    # Initialise

    Nt = int(np.round(T/dt))
    dt = T/(Nt - 1)

    if 'save_ts' in params and not params['save_ts']:
        ts = None
    else:
        ts = np.linspace(0, T, Nt)
    sim_t = 0
    us = path_handler.grid
    us_ext = path_handler.grid_ext

    th0 = path_handler.iDT(c_th)
    pi0 = path_handler.iDT(c_pi)

    if not pre_transform is None:
        th0, pi0 = pre_transform(th0, pi0)

    th_ext = path_handler.get_ext_f(th0, c_th)
    pi_ext = path_handler.get_ext_f(pi0, c_pi)

    if N_save != -1:
        if Nt % N_save != 0:
            raise Exception('Nt and N_save incompatible.')
        save_per_N = int(Nt / N_save)
        saved_ts = np.linspace(0, T, Nt)[::save_per_N]

        saved_R  = np.zeros((N_save, 3, Mm_save))
        saved_E  = np.zeros((N_save, 3, 3, Mm_save))
        saved_close_errs = np.zeros(N_save)

        if type(path_handler) == pycoss.rod.interp.cheb.ChebHandler:
            R0, E0, close_err0 = reconstruct_curve_cheb(th0, pi0, Fr_u0, Mm_render, Mm_save, path_handler, pre_transform=pre_transform_render)
        else:
            R0, E0, close_err0 = reconstruct_curve(c_th, c_pi, Fr_u0, Mm, Mm_save, path_handler_render, pre_transform=pre_transform_render)
        saved_R[0] = R0
        saved_E[0] = E0
        saved_close_errs[0] = close_err0

        if save_differential_invariants:
            saved_c_th = np.zeros((N_save, 3, Nm), dtype=c_th.dtype)
            saved_c_pi = np.zeros((N_save, 3, Nm), dtype=c_th.dtype)

            saved_th = np.zeros((N_save, 3, Mm))
            saved_pi = np.zeros((N_save, 3, Mm))
            saved_V = np.zeros((N_save, 3, Mm))
            saved_Omg = np.zeros((N_save, 3, Mm))

            saved_c_th[0] = c_th
            saved_c_pi[0] = c_pi

            saved_th[0] = th0
            saved_pi[0] = pi0
            saved_V[0] = np.nan
            saved_Omg[0] = np.nan

    time_left = -1
    t1 = time.time()

    sim_run_time = time.time()
    sim_run_time2 = 0

    # Simulate

    for sim_n in range(1, Nt):
        if verbose and sim_n % N_print == 0:
            time_passed = time.time() - t1
            t1 = time.time()
            time_per_step = time_passed / N_print
            time_left = (Nt - sim_n) * time_per_step
            time_left_hrs = time_left/60**2
            prnt_str_min_len = 50
            
            if time_left_hrs < 1:
                prnt_str = '%s%%. t=%s. Time left: %smin' % (np.round(sim_n/Nt,2)*100, np.round(sim_t,2), np.round(time_left/60, 2))
            else:
                prnt_str = '%s%%. t=%s. Time left: %sh' % (np.round(sim_n/Nt,2)*100, np.round(sim_t,2), np.round(time_left/60**2, 2))
            if len(prnt_str) < prnt_str_min_len:
                prnt_str = prnt_str + ' '*(prnt_str_min_len - len(prnt_str))
            print(prnt_str, end='\r')
        
        _sim_run_time2_timer = time.time()
        
        th, pi, V, Omg = integrator(c_th, c_pi, sim_t, dt, compute_V, compute_Omg, path_handler, lmbd, taylor_tol,
                    pre_transform=pre_transform, post_transform=post_transform)
        
        sim_run_time2 += time.time() - _sim_run_time2_timer

        th_ext = path_handler.get_ext_f(th, c_th, out=th_ext)
        pi_ext = path_handler.get_ext_f(pi, c_pi, out=pi_ext)


        sim_t += dt

        ### Reconstruct curve
        
        Fr_u0 = integrate_frame_forward(Fr_u0, V[...,0], Omg[...,0], dt)
        
        if N_save != -1 and sim_n % save_per_N == 0:
            save_k = sim_n // save_per_N
            
            if type(path_handler) == pycoss.rod.interp.cheb.ChebHandler:
                R, E, close_err = reconstruct_curve_cheb(th, pi, Fr_u0, Mm_render, Mm_save, path_handler, pre_transform=pre_transform_render)
            else:
                R, E, close_err = reconstruct_curve(c_th, c_pi, Fr_u0, Mm, Mm_save, path_handler_render, pre_transform=pre_transform_render)
            saved_R[save_k] = R
            saved_E[save_k] = E
            saved_close_errs[save_k] = close_err

            if save_differential_invariants:
                saved_c_th[save_k] = c_th
                saved_c_pi[save_k] = c_pi
                saved_th[save_k] = th
                saved_pi[save_k] = pi
                saved_V[save_k] = V
                saved_Omg[save_k] = Omg

    if type(path_handler) == pycoss.rod.interp.cheb.ChebHandler:
        Rf, Ef, close_err_f = reconstruct_curve_cheb(th, pi, Fr_u0, Mm_render, Mm_save, path_handler, pre_transform=pre_transform_render)
    else:
        Rf, Ef, close_err_f = reconstruct_curve(c_th, c_pi, Fr_u0, Mm, Mm_save, path_handler_render, pre_transform=pre_transform_render)

    sim_run_time = (time.time()-sim_run_time)/60**2
    sim_run_time2 = sim_run_time2/60**2

    res = {
        'ts' : ts,
        'Nt' : Nt,
        'us' : us,
        'us_ext' : us_ext,

        'c_th' : c_th,
        'c_pi' : c_pi,
        'Fr_u0' : Fr_u0,

        'th0' : th0,
        'pi0' : pi0,

        'th' : th,
        'pi' : pi,
        'V' : V,
        'Omg' : Omg,

        'Rf' : Rf,
        'Ef' : Ef,
        'close_err_f' : close_err_f,

        'sim_run_time' : sim_run_time,
        'sim_run_time2' : sim_run_time2,
    }

    if N_save != -1:
        res['saved_R'] = saved_R
        res['saved_E'] = saved_E
        res['saved_close_errs'] = saved_close_errs
        res['saved_ts'] = saved_ts

        if save_differential_invariants:
            res['saved_c_th'] = saved_c_th
            res['saved_c_pi'] = saved_c_pi

            res['saved_th'] = saved_th
            res['saved_pi'] = saved_pi
            res['saved_V'] = saved_V
            res['saved_Omg'] = saved_Omg

    return res
