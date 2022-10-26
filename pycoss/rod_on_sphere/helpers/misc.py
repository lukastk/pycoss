import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

import pycoss.rod.interp.FFT as FFT
import pycoss.rod.interp.cheb as cheb

def get_periodic_ext(arr):
    arr_ext_shape = list(arr.shape)
    arr_ext_shape[-1] += 1
    arr_ext = np.zeros(tuple(arr_ext_shape))
    arr_ext.T[:-1] = arr.T
    arr_ext.T[-1] = arr.T[0]
    return arr_ext

def get_arrays(shape, num, dtype=float):
    arrs = [np.zeros(shape, dtype=dtype) for i in range(num)]

    if num==1:
        return arrs[0]
    else:
        return arrs



