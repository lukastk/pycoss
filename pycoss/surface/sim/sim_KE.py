from errno import EADDRINUSE
from re import A
import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

import pycoss
from pycoss.surface.helpers.geometry import *
from pycoss.surface.helpers import integrate_frame_forward

def KE_simulate(params):
    dim = 3

    path_handler = params['path_handler']

    Nmu = path_handler.Nmu
    Nmv = path_handler.Nmv
    Lu0 = path_handler.Lu
    Lv0 = path_handler.Lv

    T = params['T']
    dt = params['dt']

    integrator = params['integrator']

    piu = np.copy(params['piu0'])
    piv = np.copy(params['piv0'])
    thu = np.copy(params['thu0'])
    thv = np.copy(params['thv0'])

    if 'Fr0_uv0' in params:
        Fr_uv0 = np.copy(params['Fr0_uv0'])
    else:
        Fr_uv0 = construct_oriented_frame(np.zeros(dim), np.eye(dim))
    
    if 'N_save' in params:
        N_save = params['N_save']
        path_handler_render = params['path_handler_render']
        Mmu_render = path_handler_render.Nmu
        Mmu_save = params['Mmu_save']
        Mmv_render = path_handler_render.Nmv
        Mmv_save = params['Mmv_save']

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

    if T/Nt != dt:
        dt = T/Nt
        print('Incompatible dt. New dt:', dt)

    if 'save_ts' in params and not params['save_ts']:
        ts = None
    else:
        ts = np.linspace(0, T, Nt+1)
        
    sim_t = 0
    #us = path_handler.grid
    #us_ext = path_handler.grid_ext

    thu0 = np.copy(thu)
    thv0 = np.copy(thv)
    piu0 = np.copy(piu)
    piv0 = np.copy(piv)

    if not pre_transform is None:
        thu0, thv0, piu0, piv0 = pre_transform(thu0, thv0, piu0, piv0)

    if N_save != -1:
        if Nt % N_save != 0:
            raise Exception('Nt and N_save incompatible.')
        save_per_N = int(Nt / N_save)
        saved_ts = np.linspace(0, T, Nt+1)[::save_per_N][1:]

        saved_R  = np.zeros((N_save, 3, Mmu_save, Mmv_save))
        saved_E  = np.zeros((N_save, 3, 3, Mmu_save, Mmv_save))
        saved_close_errs = np.zeros(N_save)

        R0, E0 = reconstruct_surface(thu0, thv0, piu0, piv0, Fr_uv0, Mmu_render, Mmv_render, Mmu_save, Mmv_save, path_handler, pre_transform=pre_transform_render)
        saved_R[0] = R0
        saved_E[0] = E0

        if save_differential_invariants:
            saved_thu = np.zeros((N_save, 3, Nmu, Nmv))
            saved_thv = np.zeros((N_save, 3, Nmu, Nmv))
            saved_piu = np.zeros((N_save, 3, Nmu, Nmv))
            saved_piv = np.zeros((N_save, 3, Nmu, Nmv))
            saved_V = np.zeros((N_save, 3, Nmu, Nmv))
            saved_Omg = np.zeros((N_save, 3, Nmu, Nmv))

            saved_thu[0] = thu0
            saved_thv[0] = thv0
            saved_piu[0] = piu0
            saved_piv[0] = piv0

    time_left = -1
    t1 = time.time()

    sim_run_time = time.time()
    sim_run_time2 = 0

    # Simulate

    X_path_handler = pycoss.surface.interp.cheb.ChebHandler((4,4), Nmu, Nmv, Lu0, Lv0, mpmath_dps=-1)
    integrability_errs = []

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
        
        thu, thv, piu, piv, V, Omg = integrator(thu, thv, piu, piv, sim_t, dt, compute_V, compute_Omg, path_handler, taylor_tol,
                    pre_transform=pre_transform, post_transform=post_transform)
        
        sim_run_time2 += time.time() - _sim_run_time2_timer

        sim_t += dt

        ### Reconstruct curve
        
        Fr_uv0 = integrate_frame_forward(Fr_uv0, V[...,0,0], Omg[...,0,0], dt)
        
        if N_save != -1 and (sim_n+1) % save_per_N == 0:
            save_k = sim_n // save_per_N
            
            if not np.allclose(sim_t, saved_ts[save_k]):
                raise Exception('Something wrong with saved_ts')

            R, E = reconstruct_surface(thu, thv, piu, piv, Fr_uv0, Mmu_render, Mmv_render, Mmu_save, Mmv_save, path_handler, pre_transform=pre_transform_render)
            saved_R[save_k] = R
            saved_E[save_k] = E

            if save_differential_invariants:
                saved_thu[save_k] = thu0
                saved_thv[save_k] = thv0
                saved_piu[save_k] = piu0
                saved_piv[save_k] = piv0


        ### Compute integrability error

        piu_hat = hat_vec_to_mat(piu)
        piv_hat = hat_vec_to_mat(piv)

        Xu = construct_se3_elem(thu, piu_hat)
        Xv = construct_se3_elem(thv, piv_hat)

        dv_Xu = X_path_handler.diffv_f(Xu)
        du_Xv = X_path_handler.diffu_f(Xv)
        ad_Xu_Xv = np.einsum('ijuv,jkuv->ikuv', Xu, Xv) - np.einsum('ijuv,jkuv->ikuv', Xv, Xu)

        integrability_err = np.max(np.abs((du_Xv + ad_Xu_Xv) - dv_Xu))
        integrability_errs.append(integrability_err)

    R, E = reconstruct_surface(thu, thv, piu, piv, Fr_uv0, Mmu_render, Mmv_render, Mmu_save, Mmv_save, path_handler, pre_transform=pre_transform_render)

    sim_run_time = (time.time()-sim_run_time)/60**2
    sim_run_time2 = sim_run_time2/60**2

    res = {
        'ts' : ts,
        'Nt' : Nt,

        'thu' : thu,
        'thv' : thv,
        'piu' : piu,
        'piv' : piv,
        'Fr_u0' : Fr_uv0,

        'thu0' : thu0,
        'thv0' : thv0,
        'piu0' : piu0,
        'piv0' : piv0,

        'V' : V,
        'Omg' : Omg,

        'Rf' : R,
        'Ef' : E,

        'sim_run_time' : sim_run_time,
        'sim_run_time2' : sim_run_time2,

        'integrability_errs' : integrability_errs
    }

    if N_save != -1:
        res['saved_R'] = saved_R
        res['saved_E'] = saved_E
        res['saved_close_errs'] = saved_close_errs
        res['saved_ts'] = saved_ts

        if save_differential_invariants:
            res['saved_thu'] = saved_thu
            res['saved_thv'] = saved_thv
            res['saved_piu'] = saved_piu
            res['saved_piv'] = saved_piv

    return res
