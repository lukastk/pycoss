import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
##import pyfftw

class UniformHandler:
    def __init__(self, shape, Nm, Mm, L, dtype=float):
    #def __init__(self, data):

        if Nm != Mm:
            raise Exception("Nm must be equal to Mm")
        
        if type(shape) == int:
            if shape == 1:
                shape = []
            else:
                shape = [shape]

        f_shape = tuple(list(shape) + [Mm])
        c_f_full_shape = tuple(list(shape) + [Mm])
        
        #f = pyfftw.empty_aligned(f_shape, dtype='float64')
        #c_f_full = pyfftw.empty_aligned(c_f_full_shape, dtype='float64')
        f = np.zeros(f_shape, dtype='float64')
        c_f_full = np.zeros(c_f_full_shape, dtype='float64')
        c_f = c_f_full[...,:Nm]

        f[:] = 0
        c_f_full[:] = 0

        self.dtype = dtype
        
        fftw_flags = ['FFTW_MEASURE', 'FFTW_DESTROY_INPUT']
        #idct_func = pyfftw.FFTW(c_f_full, f, axes=[-1], direction='FFTW_REDFT01', flags=fftw_flags)
        #dct_func = pyfftw.FFTW(f, c_f_full, axes=[-1], direction='FFTW_REDFT10', flags=fftw_flags)
        #idst_func = pyfftw.FFTW(c_f_full, f, axes=[-1], direction='FFTW_RODFT01', flags=fftw_flags)
        #dst_func = pyfftw.FFTW(f, c_f_full, axes=[-1], direction='FFTW_RODFT10', flags=fftw_flags)
        
        # This doesn't work...
        idct_func = None
        dct_func = None
        idst_func = None
        dst_func = None

        c_buffer, f_buffer = c_f, f
        c_buffer_full = c_f_full
        c_shape, f_shape = c_f.shape, f.shape
        
        ns = np.arange(Mm)
        dct_pref = np.zeros(Mm)
        dct_pref[1:] = 2
        dct_pref[0] = 1
        dct_pref *= (-1)**ns
        dct_pref /= 2*Mm
        dct_pref = dct_pref[:Nm]
        
        idct_pref = (-1)**ns
        idct_pref = idct_pref[:Nm]

        self.c_buffer = c_buffer
        self.c_buffer_full = c_buffer_full
        self.f_buffer = f_buffer
        self.dct_func = dct_func
        self.idct_func = idct_func
        self.dst_func = dst_func
        self.idst_func = idst_func
        self.c_shape = c_shape
        self.f_shape = f_shape
        self.Nm = Nm
        self.Mm = Mm
        self.L = L
        self.dct_pref = dct_pref
        self.idct_pref = idct_pref
        self.c_type = float

        self.grid = self.get_grid(self.Mm, self.L)
        self.grid_ext = self.get_grid_ext(self.Mm, self.L)
       
    def get_grid(self, N, L):
        return np.linspace(0, L, N)

    def get_grid_ext(self, N, L):
        return self.get_grid(N, L)

    def get_ext_f(self, f, c_f, out=None):
        if out is None:
            out = np.zeros(f.shape, dtype=float)

        out[:] = f
        
        return out

    def change_Mm(self, a, base_N, target_N, out=None):
        if out is None:
            out = np.zeros(a.shape, dtype=float)
        out[:] = a#*(target_N/base_N) # Do nothing
        return out

    def DT(self, fs, out=None):
        if out is None:
            out = np.zeros(self.c_shape, dtype=float)

        out[:] = fs
        
        return out

    def iDT(self, cs, out=None):
        if out is None:
            out = np.zeros(self.f_shape, dtype=float)

        out[:] = cs

        return out

    def diff_f(self, fs, out=None):
        if out is None:
            out = np.zeros(self.f_shape)

        out[:] = np.gradient(fs, self.grid, axis=-1, edge_order=2)

        return out


