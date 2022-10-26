# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
from numpy.fft import rfft, irfft, fft, ifft
from filamentsim import *
from pathlib import Path
import pickle
import timeit
import sys, os

np.seterr(all='raise')
import warnings
warnings.filterwarnings(action="error", category=np.ComplexWarning)

# ## Parameters

sim_name = 'OD_closed'
scenario_name = 'test_system'

run_name = 'test'
dt = 1e-4
integrator = 'OD_SO3_integrator'

# +
L0 = 1
L = 10
dim = 3

Nm = 100
Mm = 1024
Mm_render = Nm*100
Mm_save = Nm*10
Mm_interp = Nm*1000

# Generate curve

random_seed = 332325
frame_rot_ampl = 10
N_random_curve_modes = 3
mu_random_curve = 0
sigma_random_curve = 1

# System parameters

lmbd = 1

cn_k = np.array([1, 1, 1])*1e-2
cn_eps = np.array([1, 1, 1])*1e-2

# Simulation parameters

T = 0.01
taylor_tol = 1e-2

# Misc

N_save = 100
N_clock = 100
N_integrator_trials = int(1e4)

sim_folder = '/home/ltk26/dev/filament-dynamics/FINAL/notebooks/accuracy_sims/'
output_folder = '/home/ltk26/dev/filament-dynamics/FINAL/notebooks/accuracy_sims/output'
# -

# Load arguments if running as a script
if sys.argv[1] == "args":
    params_pkl_path = sys.argv[2]
    args_params = pickle.load(open(params_pkl_path, 'rb'))

    for param_k, param_v in args_params.items():
        globals()[param_k] = param_v

# + active=""
# if not Path(output_folder, 'systems_registry.pkl').is_file():
#     pickle.dump({}, open(Path(output_folder, 'systems_registry.pkl'), 'wb'))
#     
# system_key = (
#     'closed', L0, L, dim, Nm, Mm, Mm_render, Mm_save, Mm_interp,
#     random_seed, N_random_curve_modes, mu_random_curve, sigma_random_curve,
#     lmbd, tuple(cn_k), tuple(cn_eps),
#     T, taylor_tol,
#     N_save
# )
#
# systems_registry = pickle.load(open(Path(output_folder, 'systems_registry.pkl'), 'rb'))
#
# if not system_name in systems_registry:
#     systems_registry[system_name] = system_key
# else:
#     if system_key != systems_registry[system_name]:
#         raise Exception('System name does not match with its parameters')
#         
# pickle.dump(systems_registry, open(Path(output_folder, 'systems_registry.pkl'), 'wb'))
# -

# ## Initial conditions

# +
fft_Nm = Nm//2 + 1

us = grid(Mm, L0)
us_ext = np.linspace(0, L0, Mm+1, endpoint=True)
du = us[1] - us[0]

us_save = grid(Mm, L0)
us_save_ext = np.linspace(0, L0, Mm_save+1, endpoint=True)
du_save = us[1] - us[0]
# -

path_handler = filamentsim.FFT.FourierHandler(dim, Nm, Mm, L0)
path_handler_render = filamentsim.FFT.FourierHandler(dim, Nm, Mm_render, L0)

frame_rot_th = np.sin(2*np.pi*path_handler_render.grid/L0) * frame_rot_ampl
frame_rot_phi = np.cos(2*np.pi*path_handler_render.grid/L0) * frame_rot_ampl
frame_rot_psi = np.sin(2*np.pi*path_handler_render.grid/L0) * frame_rot_ampl

if random_seed != -1:
    np.random.seed(random_seed)


# +
def generate_random_periodic_function(dim, us, L, N, mu, sigma):
    fs = np.zeros((3, len(us)))
    dfs = np.zeros((3, len(us)))
    
    for i in range(dim):
        fs[i] += np.random.normal(mu, sigma)/2
        
        for j in range(1, N):
            a, b= np.random.normal(mu, sigma), np.random.normal(mu, sigma)
            
            fs[i] += a * np.cos((2*np.pi/L)*j*us) / (j*np.pi)
            fs[i] += b * np.sin((2*np.pi/L)*j*us) / (j*np.pi)
            
            dfs[i] += -(a * (2*np.pi/L)*j / (j*np.pi)) * np.sin((2*np.pi/L)*j*us)
            dfs[i] += (b * (2*np.pi/L)*j / (j*np.pi)) * np.cos((2*np.pi/L)*j*us)
            
    return fs, dfs

R0_raw, dR0_raw = generate_random_periodic_function(dim, path_handler_render.grid, L0,
                                       N_random_curve_modes, mu_random_curve, sigma_random_curve)

#fig = plt.figure(figsize=(10,10))
#plot_centerline(R0_raw, fig=fig)
#plt.show()

# +
c_th0, c_pi0, R_u0, E_u0, R_err0, close_err0 = get_differential_invariants_from_curve(R0_raw, Nm, Mm,
                                                                    L0, L=L, err_tol=1e-1, dR=dR0_raw)

c_th0_render = path_handler.change_Mm(c_th0, Mm, Mm_render)
c_pi0_render = path_handler.change_Mm(c_pi0, Mm, Mm_render)

th0_render = path_handler_render.iDT(c_th0_render)
pi0_render = path_handler_render.iDT(c_pi0_render)

# Rotate the frame

frame_rotation = eul2rot([frame_rot_th, frame_rot_phi, frame_rot_psi])

du_frame_rotation = np.zeros(frame_rotation.shape)
for i in range(3):
    du_frame_rotation[i,:] = path_handler_render.diff_f( frame_rotation[i,:] )
    
pi0_hat_render = hat_vec_to_mat(pi0_render)

transformed_pi0_hat_render = np.einsum('iju,kju->iku', du_frame_rotation, frame_rotation)
transformed_pi0_hat_render += np.einsum('iju,jku,lku->ilu', frame_rotation, pi0_hat_render, frame_rotation)
transformed_pi0_render = hat_mat_to_vec(transformed_pi0_hat_render)

transformed_th0_render = np.einsum('iju,ju->iu', frame_rotation, th0_render)

E_u0 = E_u0.dot(frame_rotation[...,0].T)

th0_render = transformed_th0_render
pi0_render = transformed_pi0_render

# Transform back to Fourier modes

c_th0_render = path_handler_render.DT(th0_render)
c_pi0_render = path_handler_render.DT(pi0_render)

c_th0 = path_handler.change_Mm(c_th0_render, Mm_render, Mm)
c_pi0 = path_handler.change_Mm(c_pi0_render, Mm_render, Mm)

c_th0_render = path_handler.change_Mm(c_th0, Mm, Mm_render)
c_pi0_render = path_handler.change_Mm(c_pi0, Mm, Mm_render)

th0_render = path_handler_render.iDT(c_th0_render)
pi0_render = path_handler_render.iDT(c_pi0_render)

# Plot rod

Fr0_u0 = construct_oriented_frame(R_u0, E_u0)
Fr0, R0, E0 = reconstruct_frame(th0_render, hat_vec_to_mat(pi0_render), L0, R_u0, E_u0)

fig = plt.figure(figsize=(10,10))
plot_centerline_and_frame(R0, E0, fig=fig, N_frame=10, frame_scale=0.03)
plt.show()
plt.close()

print('R_err:', R_err0)
print('close_err:', close_err0)
# -

d = 1e-3
fig, ax = plt.subplots()
ax.plot(R0[0], R0[1])
ax.plot([R0[0,-1],R0[0,0]], [R0[1,-1],R0[1,0]])
ax.set_xlim(R0[0,-1]-d, R0[0,-1]+d)
ax.set_ylim(R0[1,-1]-d, R0[1,-1]+d)
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.plot(np.abs(c_pi0[2]))
ax.set_yscale('log')
plt.show()
plt.close()


# ## Define dynamics

# +
def compute_F(t, th, pi, out=None):
    F = np.einsum('i,iu->iu', cn_k, th, out=out)
    F[0] -= cn_k[0]
    return F

def compute_M(t, th, pi, out=None):
    return np.einsum('i,iu->iu', cn_eps, pi, out=out)

def compute_U_T(us, th, pi):
    U_T = cn_k[0]*(th[0]-1)**2 + cn_k[1]*th[1]**2 + cn_k[2]*th[2]**2
    U_T = np.trapz(U_T, us)*0.5
    return U_T

def compute_U_R(us, th, pi):
    U_R = cn_eps[0]*pi[0]**2 + cn_eps[1]*pi[1]**2 + cn_eps[2]*pi[2]**2
    U_R = np.trapz(U_R, us)*0.5
    return U_R

def compute_U(us, th, pi):
    return compute_U_T(us, th, pi) + compute_U_R(us, th, pi)


# +
U0 = compute_U(path_handler_render.grid_ext,
                      path_handler_render.get_ext_f(th0_render, c_th0_render),
                      path_handler_render.get_ext_f(pi0_render, c_pi0_render))
U_T0 = compute_U_T(path_handler_render.grid_ext,
                      path_handler_render.get_ext_f(th0_render, c_th0_render),
                      path_handler_render.get_ext_f(pi0_render, c_pi0_render))
U_R0 = compute_U_R(path_handler_render.grid_ext,
                      path_handler_render.get_ext_f(th0_render, c_th0_render),
                      path_handler_render.get_ext_f(pi0_render, c_pi0_render))

print('U:', U0)
print('U_T:', U_T0)
print('U_R:', U_R0)
# -

# ## Simulate

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["NUMEXPR_NUM_THREADS"] = '1'
os.environ["OMP_NUM_THREADS"] = '1'
pyfftw.config.NUM_THREADS = 1

params = {
    'T' : T,
    'dt' : dt,
    'taylor_tol' : taylor_tol,
    
    'lmbd' : lmbd,
    
    'path_handler' : path_handler,
    'path_handler_render' : path_handler_render,
    
    'c_pi0' : c_pi0,
    'c_th0' : c_th0,
    'Fr0_u0' : Fr0_u0,
    
    'F' : compute_F,
    'M' : compute_M,
    
    'U_T' : compute_U_T,
    'U_R' : compute_U_R,
    
    'N_save' : N_save,
    'Mm_save' : Mm_save,
    
    'N_clock' : N_clock,
    
    'save_ts' : False,
}

params['integrator'] = globals()[integrator]
sim_res = OD_simulate(params)

# ## Saving results

# +
data_path = Path(output_folder, sim_name, scenario_name, run_name)
data_path.mkdir(parents=True, exist_ok=True)

figs_path = Path(data_path, 'figs')
figs_path.mkdir(parents=True, exist_ok=True)
# -

pickle.dump(sim_res, open(Path( data_path, 'sim_res.pkl' ), 'wb'))

# +
_params = dict.copy(params)

del _params['path_handler']
del _params['path_handler_render']
del _params['F']
del _params['M']
del _params['U_T']
del _params['U_R']
_params['integrator'] = integrator

pickle.dump(_params, open(Path( data_path, 'params.pkl' ), 'wb'))
# -

# ### Stats

# +
c_th = sim_res['c_th']
c_pi = sim_res['c_pi']
t = 0
dt = params['dt']
lmbd = params['lmbd']
taylor_tol = params['taylor_tol']

func = globals()[integrator]
lfunc = lambda: func(c_th, c_pi, t, dt, compute_F, compute_M, path_handler,
                 lmbd, taylor_tol, pre_transform=None, post_transform=None)
integrator_time_single_step = timeit.timeit(lfunc, number=N_integrator_trials)/N_integrator_trials
integrator_time = integrator_time_single_step * sim_res['Nt']
integrator_time /= 60**2

# +
stats = {
    'integrator_time' : integrator_time,
    'integrator_time_single_step' : integrator_time_single_step,
    'sim_run_time' : sim_res['sim_run_time'],
    'sim_run_time2' : sim_res['sim_run_time2'],
    
    'U0' : U0,
    'U_T0' : U_T0,
    'U_R0' : U_R0,
    
    'Uf' : sim_res['U'][-1],
    'U_Tf' : sim_res['U_T'][-1],
    'U_Rf' : sim_res['U_R'][-1],
}

pickle.dump(stats, open(Path( data_path, 'stats.pkl' ), 'wb'))
# -

with open(Path( data_path, 'stats.txt' ), 'w') as f:
    for k, v in stats.items():
        f.write('%s: %s\n' % (k,v))

# ### Figures

plt.rcParams['figure.facecolor'] = 'white'

fig, ax = plot_centerline_2D(sim_res['saved_R'][0])
ax.set_title('R0 vs Rf')
plot_centerline_2D(sim_res['saved_R'][-1], fig=fig, ax=ax)
plt.tight_layout()
plt.show()
fig.savefig(Path( figs_path, 'R0_vs_Rf.png' ))
plt.close()

fig, ax = plt.subplots()
ax.set_title('R0 gap')
R = R0
d = np.linalg.norm(R[:,0] - R[:,-1])*2
ax.plot(R[0], R[1])
ax.plot([R[0,-1],R[0,0]], [R[1,-1],R[1,0]])
ax.set_xlim(R[0,-1]-d, R[0,-1]+d)
ax.set_ylim(R[1,-1]-d, R[1,-1]+d)
fig.savefig(Path( figs_path, 'R0_gap.png' ))
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.set_title('Rf gap')
R = sim_res['Rf']
d = np.linalg.norm(R[:,0] - R[:,-1])*2
ax.plot(R[0], R[1])
ax.plot([R[0,-1],R[0,0]], [R[1,-1],R[1,0]])
ax.set_xlim(R[0,-1]-d, R[0,-1]+d)
ax.set_ylim(R[1,-1]-d, R[1,-1]+d)
fig.savefig(Path( figs_path, 'Rf_gap.png' ))
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.set_title('pi0 mode dist')
ax.plot(np.abs(c_pi0[0]), label='0')
ax.plot(np.abs(c_pi0[1]), label='1')
ax.plot(np.abs(c_pi0[2]), label='2')
ax.set_yscale('log')
ax.legend()
fig.savefig(Path( figs_path, 'pi0_mode_dist.png' ))
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.set_title('pi_f mode dist')
ax.plot(np.abs(sim_res['c_pi'][0]), label='0')
ax.plot(np.abs(sim_res['c_pi'][1]), label='1')
ax.plot(np.abs(sim_res['c_pi'][2]), label='2')
ax.set_yscale('log')
ax.legend()
fig.savefig(Path( figs_path, 'pi_f_mode_dist.png' ))
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.set_title('th0 mode dist')
ax.plot(np.abs(c_th0[0]), label='0')
ax.plot(np.abs(c_th0[1]), label='1')
ax.plot(np.abs(c_th0[2]), label='2')
ax.set_yscale('log')
ax.legend()
fig.savefig(Path( figs_path, 'th0_mode_dist.png' ))
plt.show()
plt.close()

fig, ax = plt.subplots()
ax.set_title('th_f mode dist')
ax.plot(np.abs(sim_res['c_th'][0]), label='0')
ax.plot(np.abs(sim_res['c_th'][1]), label='1')
ax.plot(np.abs(sim_res['c_th'][2]), label='2')
ax.set_yscale('log')
ax.legend()
fig.savefig(Path( figs_path, 'th_f_mode_dist.png' ))
plt.show()
plt.close()

# +
R = sim_res['saved_R'][0]
d = max(R[0]) - np.min(R[0])
d = max(d, np.max(R[1]) - np.min(R[1]))
d = max(d, np.max(R[2]) - np.min(R[2]))
frame_scale = d/20
N_frame = 15

data_path = Path(output_folder, scenario_name, sim_name)
vid_centerline_and_frame_3D(sim_res['saved_R'], sim_res['saved_E'],
                            save_name='%s/video.mp4' % data_path, frame_scale=frame_scale, N_frame=N_frame)
plt.show()
plt.close()
