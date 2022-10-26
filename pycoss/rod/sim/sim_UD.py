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

def UD_simulate(params):
    dim = 3

    path_handler = params['path_handler']

    Nm = path_handler.Nm
    Mm = path_handler.Mm
    L0 = path_handler.L

    T = params['T']
    dt = params['dt']

    alpha = params['alpha']
    mI = params['mI']
    lmbd = params['lmbd']
    imI = np.linalg.inv(mI)

    integrator = params['integrator']

    c_pi = np.copy(params['c_pi0'])
    c_th = np.copy(params['c_th0'])

    if 'Fr0_u0' in params:
        Fr_u0 = np.copy(params['Fr0_u0'])
    else:
        Fr_u0 = construct_oriented_frame(np.zeros(dim), np.eye(dim))
    
    if 'c_V0' in params:
        c_V = np.copy(params['c_V0'])
    else:
        c_V = np.zeros(c_pi.shape, dtype=path_handler.c_type)
    if 'c_Omg0' in params:
        c_Omg = np.copy(params['c_Omg0'])
    else:
        c_Omg = np.zeros(c_pi.shape, dtype=path_handler.c_type)

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

    compute_F = params['F']
    compute_M = params['M']

    if 'U_T' in params:
        compute_U_T = params['U_T']
        compute_U_R = params['U_R']
        compute_K_T = params['K_T']
        compute_K_R = params['K_R']

        compute_energy = True
    else:
        compute_energy = False

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

    if T/Nt != dt:
        dt = T/Nt
        print('Incompatible dt. New dt:', dt)

    if 'save_ts' in params and not params['save_ts']:
        ts = None
    else:
        ts = np.linspace(0, T, Nt+1)

    sim_t = 0
    us = path_handler.grid
    us_ext = path_handler.grid_ext

    th0 = path_handler.iDT(c_th)
    pi0 = path_handler.iDT(c_pi)
    V0 = path_handler.iDT(c_V)
    Omg0 = path_handler.iDT(c_Omg)

    if not pre_transform is None:
        th0, pi0, V0, Omg0 = pre_transform(th0, pi0, V0, Omg0)

    U_Ts = np.zeros(Nt)
    U_Rs = np.zeros(Nt)
    Us = np.zeros(Nt)
    K_Ts = np.zeros(Nt)
    K_Rs = np.zeros(Nt)
    Ks = np.zeros(Nt)
    Es = np.zeros(Nt)

    th_ext = path_handler.get_ext_f(th0, c_th)
    pi_ext = path_handler.get_ext_f(pi0, c_pi)
    V_ext = path_handler.get_ext_f(V0, c_V)
    Omg_ext = path_handler.get_ext_f(Omg0, c_Omg)

    if compute_energy:
        U_Ts[0] = compute_U_T(us_ext, th_ext, pi_ext)
        U_Rs[0] = compute_U_R(us_ext, th_ext, pi_ext)
        Us[0] = U_Ts[0] + U_Rs[0]
        K_Ts[0] = compute_K_T(us_ext, V_ext, Omg_ext, mI)
        K_Rs[0] = compute_K_R(us_ext, V_ext, Omg_ext, mI)
        Ks[0] = K_Ts[0] + K_Rs[0]
        Es[0] = Us[0] + Ks[0]

    if N_save != -1:
        if Nt % N_save != 0:
            raise Exception('Nt and N_save incompatible.')
        save_per_N = int(Nt / N_save)
        saved_ts = np.linspace(0, T, Nt+1)[::save_per_N][1:]

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
            saved_c_V = np.zeros((N_save, 3, Nm), dtype=c_th.dtype)
            saved_c_Omg = np.zeros((N_save, 3, Nm), dtype=c_th.dtype)

            saved_th = np.zeros((N_save, 3, Mm))
            saved_pi = np.zeros((N_save, 3, Mm))
            saved_V = np.zeros((N_save, 3, Mm))
            saved_Omg = np.zeros((N_save, 3, Mm))

            saved_c_th[0] = c_th
            saved_c_pi[0] = c_pi
            saved_c_V[0] = c_V
            saved_c_Omg[0] = c_Omg

            saved_th[0] = th0
            saved_pi[0] = pi0
            saved_V[0] = V0
            saved_Omg[0] = Omg0

    time_left = -1
    t1 = time.time()

    sim_run_time = time.time()
    sim_run_time2 = 0

    # Simulate

    for sim_n in range(0, Nt):
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
        
        th, pi, V, Omg, L, F, M = integrator(c_th, c_pi, c_V, c_Omg, sim_t, dt, compute_F, compute_M, path_handler,
            alpha, mI, imI, lmbd, taylor_tol, pre_transform=pre_transform, post_transform=post_transform)
        
        sim_run_time2 += time.time() - _sim_run_time2_timer
        
        th_ext = path_handler.get_ext_f(th, c_th, out=th_ext)
        pi_ext = path_handler.get_ext_f(pi, c_pi, out=pi_ext)
        V_ext = path_handler.get_ext_f(V, c_V, out=V_ext)
        Omg_ext = path_handler.get_ext_f(Omg, c_Omg, out=Omg_ext)
        
        if compute_energy:
            U_Ts[sim_n] = compute_U_T(us_ext, th_ext, pi_ext)
            U_Rs[sim_n] = compute_U_R(us_ext, th_ext, pi_ext)
            Us[sim_n] = U_Ts[sim_n] + U_Rs[sim_n]
            K_Ts[sim_n] = compute_K_T(us_ext, V_ext, Omg_ext, mI)
            K_Rs[sim_n] = compute_K_R(us_ext, V_ext, Omg_ext, mI)
            Ks[sim_n] = K_Ts[sim_n] + K_Rs[sim_n]
            Es[sim_n] = Us[sim_n] + Ks[sim_n]

        sim_t += dt

        ### Reconstruct curve
        
        Fr_u0 = integrate_frame_forward(Fr_u0, V[...,0], Omg[...,0], dt)
        
        if N_save != -1 and (sim_n+1) % save_per_N == 0:
            save_k = sim_n // save_per_N

            if not np.allclose(sim_t, saved_ts[save_k]):
                raise Exception('Something wrong with saved_ts')
            
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
                saved_c_V[save_k] = c_V
                saved_c_Omg[save_k] = c_Omg

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
        'dt' : dt,
        'Nt' : Nt,
        'ts' : ts,
        'us' : us,
        'us_ext' : us_ext,

        'c_th' : c_th,
        'c_pi' : c_pi,
        'c_V' : c_V,
        'c_Omg' : c_Omg,
        'Fr_u0' : Fr_u0,

        'th' : th,
        'pi' : pi,
        'V' : V,
        'Omg' : Omg,
        'L' : L,

        'U_T' : U_Ts,
        'U_R' : U_Rs,
        'U' : Us,
        'K_T' : K_Ts,
        'K_R' : K_Rs,
        'K' : Ks,
        'E' : Es,

        'Rf' : Rf,
        'Ef' : Ef,
        'close_err_f' : close_err_f,

        'sim_run_time' : sim_run_time,
        'sim_run_time2' : sim_run_time2,
    }

    if N_save != -1:
        res['saved_R'] = saved_R
        res['saved_E'] = saved_E

        if save_differential_invariants:
            
            res['saved_c_th'] = saved_c_th
            res['saved_c_pi'] = saved_c_pi
            res['saved_c_V'] = saved_c_V
            res['saved_c_Omg'] = saved_c_Omg

            res['saved_th'] = saved_th
            res['saved_pi'] = saved_pi
            res['saved_V'] = saved_V
            res['saved_Omg'] = saved_Omg

        res['saved_close_errs'] = saved_close_errs
        res['saved_ts'] = saved_ts

    return res


