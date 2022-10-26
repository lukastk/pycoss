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
    def __init__(self, shape, Nmu, Nmv, Lu, Lv, mpmath_dps=100, dtype=float):
        
        if type(shape) == int:
            if shape == 1:
                shape = []
            else:
                shape = [shape]

        f_shape = tuple(list(shape) + [Nmu, Nmv])

        f = np.zeros(f_shape, dtype=dtype)

        self.dtype = dtype
        
        # This doesn't work...
        idct_func = None
        dct_func = None
        idst_func = None
        dst_func = None

        f_buffer = f
        f_shape = f.shape

        self.f_buffer = f_buffer
        self.dct_func = dct_func
        self.idct_func = idct_func
        self.dst_func = dst_func
        self.idst_func = idst_func
        self.f_shape = f_shape
        self.Nmu = Nmu
        self.Nmv = Nmv
        self.Lu = Lu
        self.Lv = Lv
        self.mpmath_dps = mpmath_dps

        self.gridu = get_grid_1D(self.Nmu, self.Lu)
        self.gridv = get_grid_1D(self.Nmv, self.Lv)
        self.grid = get_grid_2D(self.Nmu, self.Nmv, self.Lu, self.Lv)

        self.cheb_Dus = []
        self.cheb_Dus.append(self.get_cheb_D(self.Nmu, self.Lu, xmin=0, order=1, mpmath_dps=self.mpmath_dps))
        self.cheb_Dus.append(self.get_cheb_D(self.Nmu, self.Lu, xmin=0, order=2, mpmath_dps=self.mpmath_dps))

        self.cheb_Dvs = []
        self.cheb_Dvs.append(self.get_cheb_D(self.Nmv, self.Lv, xmin=0, order=1, mpmath_dps=self.mpmath_dps))
        self.cheb_Dvs.append(self.get_cheb_D(self.Nmv, self.Lv, xmin=0, order=2, mpmath_dps=self.mpmath_dps))
        
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

    def cheb_to_unif(self, fs, Mmu_unif=None, Mmv_unif=None):
        if Mmu_unif is None:
            Mmu_unif = self.Nmu
        if Mmv_unif is None:
            Mmv_unif = self.Nmv
        
        gridu_unif = self.get_grid_unif(Mmu_unif, self.Lu)
        gridv_unif = self.get_grid_unif(Mmv_unif, self.Lv)

        #return scipy.interpolate.barycentric_interpolate(self.grid, fs, grid_unif, axis=-1)
        return cheb_interpolate_2D(fs, gridu_unif, gridv_unif, umin=0, vmin=0, Lu=self.Lu, Lv=self.Lv)


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
        return cheb_interpolate_2D(fs, grid2, xmin=0, L=self.L)

    def get_grid_unif(self, Mm, L):
        return np.array(np.linspace(0, L, Mm), dtype=self.dtype)

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

    def diffu_f(self, fs, out=None, order=1):
        if out is None:
            out = np.zeros(fs.shape, dtype=self.dtype)

        if order > len(self.cheb_Dus):
            for i in range(len(self.cheb_Dus), order):
                self.cheb_Dus.append(self.get_cheb_D(self.Nmu, self.Lu, xmin=0, order=i+1, mpmath_dps=self.mpmath_dps))

        np.einsum('ij,...jk->...ik', self.cheb_Dus[order-1], fs, out=out)
        #np.dot(self.cheb_Ds[order-1], fs, out=out)

        return out

    def diffv_f(self, fs, out=None, order=1):
        if out is None:
            out = np.zeros(fs.shape, dtype=self.dtype)

        if order > len(self.cheb_Dvs):
            for i in range(len(self.cheb_Dvs), order):
                self.cheb_Dvs.append(self.get_cheb_D(self.Nmv, self.Lv, xmin=0, order=i+1, mpmath_dps=self.mpmath_dps))

        np.einsum('ij,...kj->...ki', self.cheb_Dvs[order-1], fs, out=out)
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

def cheb_interpolate_1D(fs, xs, xmin=0, L=1, out=None):
    xs = 2*(xs-xmin)/L - 1
    
    n = fs.shape[-1] - 1
    cts, ws = chebyshev2(n)

    if out is None:
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

def cheb_interpolate_2D(fs, us, vs, umin=0, vmin=0, Lu=1, Lv=1):
    nu = fs.shape[-2] - 1
    out1 = np.zeros( tuple( list(fs.shape[:-2]) + [nu, len(vs)] ) )

    for i in range(nu):
        cheb_interpolate_1D(fs[...,i,:], vs, xmin=vmin, L=Lv, out=out1[...,i,:])

    out2 = np.zeros( tuple( list(fs.shape[:-2]) + [len(us), len(vs)] ) )

    for i in range(len(vs)):
        cheb_interpolate_1D(out1[...,:,i], us, xmin=umin, L=Lu, out=out2[...,:,i])

    return out2

def get_grid_1D(Mm, L):
    N = Mm - 1
    x = np.cos(np.pi*np.arange(0,N+1)/N)
    x = L*(1-x)/2
    return x

def get_grid_2D(Mmu, Mmv, Lu, Lv):
    us = get_grid_1D(Mmu, Lu)
    vs = get_grid_1D(Mmv, Lv)
    return np.meshgrid(us, vs, indexing='ij')