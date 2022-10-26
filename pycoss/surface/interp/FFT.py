import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
#import pyfftw

class FourierHandler:

    def __init__(self, shape, Nm, Mm, L, dtype=float):
        if type(shape) == int:
            if shape == 1:
                shape = []
            else:
                shape = [shape]
            
        if Mm//2+1 < Nm:
            raise Exception('Mm too small.')

        f_shape = tuple(list(shape) + [Mm])
        c_f_full_shape = tuple(list(shape) + [Mm//2 + 1])
        
        #f = pyfftw.empty_aligned(f_shape, dtype='float64')
        #c_f_full = pyfftw.empty_aligned(c_f_full_shape, dtype='complex128')
        f = np.zeros(f_shape, dtype='float64')
        c_f_full = np.zeros(c_f_full_shape, dtype='complex128')
        c_f = c_f_full[...,:Nm]

        f[:] = 0
        c_f_full[:] = 0

        self.dtype = dtype
        
        fftw_flags = ['FFTW_MEASURE', 'FFTW_DESTROY_INPUT']
        #irfft_func = pyfftw.FFTW(c_f_full, f, axes=[-1], direction='FFTW_BACKWARD', flags=fftw_flags)
        #rfft_func = pyfftw.FFTW(f, c_f_full, axes=[-1], direction='FFTW_FORWARD', flags=fftw_flags)
        
        c_buffer, f_buffer = c_f, f
        c_buffer_full = c_f_full
        c_shape, f_shape = c_f.shape, f.shape

        self.c_buffer = c_buffer
        self.c_buffer_full = c_buffer_full
        self.f_buffer = f_buffer
        #self.rfft_func = rfft_func
        #self.irfft_func = irfft_func
        self.c_shape = c_shape
        self.f_shape = f_shape
        self.Nm = Nm
        self.Mm = Mm
        self.L = L
        self.c_type = complex

        self.grid = self.get_grid(self.Mm, self.L)
        self.grid_ext = self.get_grid_ext(self.Mm, self.L)
        self.du = self.get_du(self.Mm, self.L)

        self.FFT_dws = []
        for di in range(10):
            self.FFT_dws.append(  ( (2*np.pi * 1j / self.L) * np.arange(0, self.Nm) )**di  )

    def get_grid(self, N, L):
        return np.linspace(0, L, N, endpoint=False)

    def get_grid_ext(self, N, L):
        return np.linspace(0, L, N+1, endpoint=True)

    def get_du(self, N, L):
        return L/N

    def get_ext_f(self, f, c_f, out=None):
        if out is None:
            out = np.zeros(f.shape[:-1] + (f.shape[-1]+1,), dtype=float)
        out[...,:-1] = f
        out[...,-1] = f[...,0]

        return out

    def DT(self, f, out=None):
        
        if out is None:
            out = np.zeros(self.c_shape, dtype=complex)

        Nm = self.Nm

        #self.f_buffer[:] = f
        #out[:] = self.rfft_func()[...,:Nm]
        
        out[:] = scipy.fft.rfft(f)[...,:Nm]

        return out
        
    def iDT(self, c_f, out=None):    

        if out is None:
            out = np.zeros(self.f_shape)
        
        Nm = self.Nm
        c_buffer_full = self.c_buffer_full

        c_buffer_full[...,:Nm] = c_f
        c_buffer_full[...,Nm:] = 0
        #out[:] = self.irfft_func()

        out[:] = scipy.fft.irfft(c_buffer_full)

        return out
        
    def diff_f(self, f, out=None, deriv_order=1):

        if out is None:
            out = np.zeros(self.f_shape)
            
        c_buffer = self.c_buffer

        self.DT(f, out=c_buffer)
        self.diff_cf(c_buffer, out=c_buffer, deriv_order=deriv_order)
        self.iDT(c_buffer, out=out)
        
        return out

    def int_f(self, f, out=None, int_order=1):
        if out is None:
            out = np.zeros(self.f_shape)

        c_buffer = self.c_buffer
            
        self.DT(f, out=c_buffer)
        self.int_cf(c_buffer, out=c_buffer, int_order=int_order)
        self.iDT(c_buffer, out=out)
        
        return out

    def diff_cf(self, c_a, out=None, deriv_order=1):  
        if out is None:
            out = np.zeros(self.c_shape, dtype=complex)
            
        Nm = self.Nm
        L = self.L

        dw = self.FFT_dws[deriv_order]
        out[:] = c_a
        out *= dw
        
        if deriv_order % 2 != 0 and Nm % 2 == 0:
            out[...,-1] = 0 # Set component corresponding to the Nyquist frequency to zero
            
        return out

    def int_cf(self, c_a, out=None, int_order=1):
        if out is None:
            out = np.zeros(self.c_shape, dtype=complex)

        Nm = self.Nm
        L = self.L

        dw = get_FFT_diff_dw(Nm, int_order, L)
        out[:] = c_a
        out[...,1:] /= dw[...,1:]
        out[0] = 0
                
        return out


    def change_Mm(self, a, base_N, target_N, out=None):
        if out is None:
            out = np.zeros(a.shape, dtype=complex)

        out[:] = a*(target_N/base_N)
        return out





def grid(N, L):
    return np.linspace(0, L, N, endpoint=False)

def grid_ext(N, L):
    return np.linspace(0, L, N+1, endpoint=True)

def get_du(N, L):
    return L/N

FFT_diff_dws = {}
def get_FFT_diff_dw(fft_Nm, di, L):
    if not (fft_Nm, di, L) in FFT_diff_dws:
        FFT_diff_dws[(fft_Nm, di, L)] = ( (2*np.pi * 1j / L) * np.arange(0, fft_Nm) )**di
    return FFT_diff_dws[(fft_Nm, di, L)]