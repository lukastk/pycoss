import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time
import scipy.interpolate
import mpmath
##import pyfftw

class ChebHandler:
    def __init__(self, shape, Nm, Mm, L, mpmath_dps=100, dtype=float):
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

        f = np.zeros(f_shape, dtype=dtype)
        c_f_full = np.zeros(c_f_full_shape, dtype=dtype)
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
        self.dgrid = self.get_dgrid(self.Mm, self.L)
        self.grid_ext = self.get_grid_ext(self.Mm, self.L)

        self.cheb_Ds = []
        self.mpmath_dps = mpmath_dps
        self.cheb_Ds.append(self.get_cheb_D(self.Mm, self.L, xmin=0, order=1, mpmath_dps=self.mpmath_dps))
        self.cheb_Ds.append(self.get_cheb_D(self.Mm, self.L, xmin=0, order=2, mpmath_dps=self.mpmath_dps))

        #self.cheb_Ds = self.get_cheb_D(self.Mm, self.L, xmin=0, order=[1,2], mpmath_dps=self.mpmath_dps)
        
    def get_cheb_D(self, Mm, L, xmin=0, order=1, mpmath_dps=100):
        '''Chebushev polynomial differentiation matrix.
        Ref.: Trefethen's 'Spectral Methods in MATLAB' book.
        '''

        N = Mm - 1

        if mpmath_dps != -1:
            mpmath.mp.dps = mpmath_dps
            mpmath_num = mpmath.mpf(1)
        else:
            mpmath_num = self.dtype(1)

        x      = mpmath_num*np.cos(np.pi*np.arange(0,N+1)/N)
        if N%2 == 0:
            x[N//2] = 0.0 # only when N is even!
        c      = mpmath_num*np.ones(N+1); c[0] = 2.0; c[N] = 2.0
        c      = c * (-1.0)**np.arange(0,N+1)
        c      = c.reshape(N+1,1)
        X      = np.tile(x.reshape(N+1,1), (1,N+1))
        dX     = X - X.T
        D      = np.dot(c, 1.0/c.T) / (dX+np.eye(N+1))
        D      = D - np.diag( D.sum(axis=1) )
        
        
        # Change the domain
        x = xmin + L*(1-x)/2
        D = -D[:,:] / (L/2)

        for i in range(order-1):
            D = D.dot(D)

        D = np.array(D, dtype=self.dtype)
        x = np.array(x, dtype=self.dtype)
        
        return D

    def cheb_to_unif(self, fs, Mm_unif=None):
        if Mm_unif is None:
            Mm_unif = self.Mm

        grid_unif = self.get_grid_unif(Mm_unif, self.L)

        #return scipy.interpolate.barycentric_interpolate(self.grid, fs, grid_unif, axis=-1)
        return cheb_interpolate(fs, grid_unif, xmin=0, L=self.L)


    def unif_to_cheb(self, fs, Mm_cheb=None):
        if Mm_cheb is None:
            Mm_cheb = self.Mm

        grid_unif = self.get_grid_unif(fs.shape[-1], self.L)
        grid = self.get_grid(Mm_cheb, self.L)

        return scipy.interpolate.barycentric_interpolate(grid_unif, fs, grid, axis=-1)

    def cheb_to_cheb(self, fs, Mm2=None):
        if Mm2 is None:
            Mm2 = self.Mm

        grid1 = self.get_grid(fs.shape[-1], self.L)
        grid2 = self.get_grid(Mm2, self.L)

        #return scipy.interpolate.barycentric_interpolate(grid1, fs, grid2, axis=-1)
        return cheb_interpolate(fs, grid2, xmin=0, L=self.L)

    def get_grid(self, Mm, L):
        N = Mm - 1
        x = np.cos(self.dtype(np.pi)*np.arange(0,N+1)/N)
        x = self.L*(1-x)/2
        return x

    def get_dgrid(self, Mm, L):
        grid = self.get_grid(Mm, L)
        dgrid = grid[1:] - grid[:-1]
        return dgrid

    def get_grid_unif(self, Mm, L):
        return np.array(np.linspace(0, L, Mm), dtype=self.dtype)

    def get_grid_ext(self, N, L):
        return self.get_grid(N, L)

    def get_ext_f(self, f, c_f, out=None):
        if out is None:
            out = np.zeros(f.shape, dtype=self.dtype)

        out[:] = f
        
        return out

    def change_Mm(self, a, base_N, target_N, out=None):
        if out is None:
            out = np.zeros(a.shape, dtype=self.dtype)
        out[:] = a#*(target_N/base_N) # Do nothing
        return out

    def DT(self, fs, out=None):
        if out is None:
            out = np.zeros(self.c_shape, dtype=self.dtype)

        out[:] = fs
        
        return out

    def iDT(self, cs, out=None):
        if out is None:
            out = np.zeros(self.f_shape, dtype=self.dtype)

        out[:] = cs

        return out

    def diff_f(self, fs, out=None, order=1):
        if out is None:
            out = np.zeros(fs.shape, dtype=self.dtype)

        if order > len(self.cheb_Ds):
            for i in range(len(self.cheb_Ds), order):
                self.cheb_Ds.append(self.get_cheb_D(self.Mm, self.L, xmin=0, order=i+1, mpmath_dps=self.mpmath_dps))

        np.einsum('ij,...j->...i', self.cheb_Ds[order-1], fs, out=out)
        #np.dot(self.cheb_Ds[order-1], fs, out=out)

        return out


def barycentric_weights(n):
    j = np.arange(0, n+1)
    w = np.power(-1, j)
    o = np.ones(n+1)
    o[0] = 0.5
    o[n] = 0.5
    w = np.flipud(w*o)
    return w

def chebyshev_nodes(n):
    ts = -np.cos(np.pi * np.arange(n+1) / (n)) # Minus sign here is so that the the points are in increasing order
    return ts

def chebyshev2(n):
    collocation_ts = chebyshev_nodes(n)
    collocation_w = barycentric_weights(n)
    return collocation_ts, collocation_w

def cheb_interpolate(fs, xs, xmin=0, L=1):
    xs = 2*(xs-xmin)/L - 1
    
    n = fs.shape[-1] - 1
    cts, ws = chebyshev2(n)
    out = np.zeros( tuple( list(fs.shape[:-1]) + [len(xs)] ) )

    fsws = fs*ws

    for j, x in enumerate(xs):
        dx = x-cts
        dx0 = np.argwhere(dx == 0)

        if len(dx0) > 0:
            out[...,j] = fs[...,dx0[0,0]]
        else:
            A = np.sum( fsws/dx , axis=-1)
            B = np.sum( ws/dx , axis=-1)
            out[...,j] = A/B

    return out