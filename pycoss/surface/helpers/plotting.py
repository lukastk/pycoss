import scipy
import scipy.fftpack
import scipy.special
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import proj3d

def set_aspect_equal(ax):
    """ 
    Fix the 3D graph to have similar scale on all the axes.
    Call this after you do all the plot3D, but before show
    """
    X = ax.get_xlim3d()
    Y = ax.get_ylim3d()
    Z = ax.get_zlim3d()
    a = [X[1]-X[0],Y[1]-Y[0],Z[1]-Z[0]]
    b = np.amax(a)
    ax.set_xlim3d(X[0]-(b-a[0])/2,X[1]+(b-a[0])/2)
    ax.set_ylim3d(Y[0]-(b-a[1])/2,Y[1]+(b-a[1])/2)
    ax.set_zlim3d(Z[0]-(b-a[2])/2,Z[1]+(b-a[2])/2)
    ax.set_box_aspect(aspect = (1,1,1))

def orthogonal_proj(zfront, zback):
    a = (zfront+zback)/(zfront-zback)
    b = -2*(zfront*zback)/(zfront-zback)
    return np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,a,b],
                        [0,0,0,zback]])

def plot_surface(R, E, shell_width, fig=None, ax=None, N_frame=15, frame_scale=0.1, from_top=False, perspective=True):
    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = plt.axes(projection='3d')

    R2 = R + E[:,0,:,:]*shell_width

    stride=2
    ax.plot_surface(R[0], R[1], R[2], linewidth=0.5, cstride=stride, rstride=stride, color=(0,0.4,0,0.3))
    ax.plot_wireframe(R[0], R[1], R[2], linewidth=0.5, cstride=stride, rstride=stride, color=(0,0,0,0.3))

    ax.plot_surface(R2[0], R2[1], R2[2], linewidth=0.5, cstride=stride, rstride=stride, color=(0,0,0.4,0.3))

    set_aspect_equal(ax)

    if from_top:
        ax.view_init(azim=0, elev=90)

    if not perspective:
        proj3d.persp_transformation = orthogonal_proj

    return fig, ax

def vid_surface(Rs, Es, shell_width, fig=None, ax=None, save_name=None, frame_scale=0.1, N_frame=15, from_top=False, perspective=True):
    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = plt.axes(projection='3d')

    R = Rs[0]
    E = Es[0]
    R2 = R + E[:,0,:,:]*shell_width

    stride=2
    g_R1 = ax.plot_surface(R[0], R[1], R[2], linewidth=0.5, cstride=stride, rstride=stride, color=(0,0.4,0,0.3))
    g_R1_mesh = ax.plot_wireframe(R[0], R[1], R[2], linewidth=0.5, cstride=stride, rstride=stride, color=(0,0,0,0.3))
    g_R2 = ax.plot_surface(R2[0], R2[1], R2[2], linewidth=0.5, cstride=stride, rstride=stride, color=(0,0,0.4,0.3))

    plots = [g_R1, g_R1_mesh, g_R2]

    if from_top:
        ax.view_init(azim=0, elev=90)

    if not perspective:
        proj3d.persp_transformation = orthogonal_proj
    
    set_aspect_equal(ax)


    def animate(k):
        R = Rs[k]
        E = Es[k]
        R2 = R + E[:,0,:,:]*shell_width
        
        plots[0].remove()
        plots[1].remove()
        plots[2].remove()
        
        g_R1 = ax.plot_surface(R[0], R[1], R[2], linewidth=0.5, cstride=stride, rstride=stride, color=(0,0.4,0,0.3))
        g_R1_mesh = ax.plot_wireframe(R[0], R[1], R[2], linewidth=0.5, cstride=stride, rstride=stride, color=(0,0,0,0.3))
        g_R2 = ax.plot_surface(R2[0], R2[1], R2[2], linewidth=0.5, cstride=stride, rstride=stride, color=(0,0,0.4,0.3))

        plots.clear()
        plots.append(g_R1)
        plots.append(g_R1_mesh)
        plots.append(g_R2)
        
        return g_R1,

    ani = animation.FuncAnimation(
        fig, animate, interval=1, blit=True, save_count=Rs.shape[0])

    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    
    if not save_name is None:
        ani.save(save_name, writer=writer)

    return fig, ani