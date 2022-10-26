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

def plot_centerline_2D(R, fig=None, ax=None):

    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = plt.axes()

    ax.plot(R[0], R[1])

    ax.set_aspect('equal')

    return fig, ax

def plot_centerline(R, fig=None, ax=None, from_top=False, perspective=True):

    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = plt.axes(projection='3d')

    ax.plot3D(R[0], R[1], R[2])

    set_aspect_equal(ax)

    if from_top:
        ax.view_init(azim=0, elev=90)

    if not perspective:
        proj3d.persp_transformation = orthogonal_proj

    return fig, ax

def plot_centerline_and_frame(R, E, fig=None, ax=None, N_frame=15, frame_scale=0.1, from_top=False, perspective=True):
    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = plt.axes(projection='3d')

    e1, e2, e3 = E[:,0], E[:,1], E[:,2]

    ax.plot3D(R[0], R[1], R[2])

    Mm = R.shape[-1]

    frame_scale = frame_scale * (np.max(R) - np.min(R))
    for i in range(0, Mm, int(Mm/N_frame)):
        _rx, _ry, _rz = R[:,i]
        dx,dy,dz = e1[:,i]*frame_scale
        ax.plot3D([_rx,_rx+dx], [_ry,_ry+dy], [_rz,_rz+dz], color='blue')
        
        _rx, _ry, _rz = R[:,i]
        dx,dy,dz = e2[:,i]*frame_scale
        ax.plot3D([_rx,_rx+dx], [_ry,_ry+dy], [_rz,_rz+dz], color='green')
        
        _rx, _ry, _rz = R[:,i]
        dx,dy,dz = e3[:,i]*frame_scale
        ax.plot3D([_rx,_rx+dx], [_ry,_ry+dy], [_rz,_rz+dz], color='red')
        
    set_aspect_equal(ax)

    if from_top:
        ax.view_init(azim=0, elev=90)

    if not perspective:
        proj3d.persp_transformation = orthogonal_proj

    return fig, ax

def plot_centerline_and_frame_2D(R, E, fig=None, ax=None, N_frame=15, frame_scale=0.1):

    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = plt.axes()

    ax.plot(R[0], R[1])

    e1, e2, e3 = E[:,0], E[:,1], E[:,2]

    Mm = R.shape[-1]

    frame_scale = frame_scale * (np.max(R) - np.min(R))
    for i in range(0, Mm, int(Mm/N_frame)):
        _rx, _ry, _rz = R[:,i]
        dx,dy,dz = e1[:,i]*frame_scale
        ax.plot([_rx,_rx+dx], [_ry,_ry+dy], color='blue')
        
        _rx, _ry, _rz = R[:,i]
        dx,dy,dz = e2[:,i]*frame_scale
        ax.plot([_rx,_rx+dx], [_ry,_ry+dy], color='green')

    ax.set_aspect('equal')

    return fig, ax

def vid_centerline_2D(Rs, fig=None, ax=None, save_name=None):
    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = plt.axes()

    R = Rs[0]

    g_R, = ax.plot(R[0], R[1])

    xmin, xmax = np.min(Rs[:,0]), np.max(Rs[:,0])
    ymin, ymax = np.min(Rs[:,1]), np.max(Rs[:,1])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    def animate(i):
        R = Rs[i]
        
        g_R.set_xdata(R[0])
        g_R.set_ydata(R[1])

        return g_R,

    ani = animation.FuncAnimation(
        fig, animate, interval=1, blit=True, save_count=Rs.shape[0])

    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    if not save_name is None:
        ani.save(save_name, writer=writer)

    return ani

def vid_centerline_and_frame_2D(Rs, Es, fig=None, ax=None, save_name=None, frame_scale=0.1, N_frame=15,):
    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = plt.axes()

    R = Rs[0]
    E = Es[0]

    e1, e2, e3 = E[:,0], E[:,1], E[:,2]

    g_R, = ax.plot(R[0], R[1])

    g_Es = []

    for i in range(0, R.shape[-1], int(R.shape[-1]/N_frame)):
        _rx, _ry, _rz = R[:,i]
        dx,dy,dz = e1[:,i]*frame_scale
        g_e1, = ax.plot([_rx,_rx+dx], [_ry,_ry+dy], color='blue')

        dx,dy,dz = e2[:,i]*frame_scale
        g_e2, = ax.plot([_rx,_rx+dx], [_ry,_ry+dy], color='green')
        
        g_Es.append((g_e1, g_e2))

    xmin, xmax = np.min(Rs[:,0]), np.max(Rs[:,0])
    ymin, ymax = np.min(Rs[:,1]), np.max(Rs[:,1])

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    def animate(k):
        R = Rs[k]
        E = Es[k]
        
        g_R.set_xdata(R[0])
        g_R.set_ydata(R[1])
        
        e1, e2, e3 = E[:,0], E[:,1], E[:,2]

        l = 0
        for i in range(0, R.shape[-1], int(R.shape[-1]/N_frame)):
            g_e1, g_e2 = g_Es[l]
            l += 1
            
            _rx, _ry, _rz = R[:,i]
            dx,dy,dz = e1[:,i]*frame_scale
            g_e1.set_xdata([_rx,_rx+dx])
            g_e1.set_ydata([_ry,_ry+dy])

            dx,dy,dz = e2[:,i]*frame_scale
            g_e2.set_xdata([_rx,_rx+dx])
            g_e2.set_ydata([_ry,_ry+dy])
        
        return g_R,

    ani = animation.FuncAnimation(
        fig, animate, interval=1, blit=True, save_count=Rs.shape[0])

    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    if not save_name is None:
        ani.save(save_name, writer=writer)

    return ani

def vid_centerline_and_frame_3D(Rs, Es, fig=None, ax=None, save_name=None, frame_scale=0.1, N_frame=15, from_top=False, perspective=True):
    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = plt.axes(projection='3d')

    R = Rs[0]
    E = Es[0]

    e1, e2, e3 = E[:,0], E[:,1], E[:,2]

    g_R, = ax.plot3D(R[0], R[1], R[2])

    g_Es = []

    if from_top:
        ax.view_init(azim=0, elev=90)

    if not perspective:
        proj3d.persp_transformation = orthogonal_proj

    for i in range(0, R.shape[-1], int(R.shape[-1]/N_frame)):
        _rx, _ry, _rz = R[:,i]
        dx,dy,dz = e1[:,i]*frame_scale
        g_e1, = ax.plot3D([_rx,_rx+dx], [_ry,_ry+dy], [_rz,_rz+dz], color='blue')

        dx,dy,dz = e2[:,i]*frame_scale
        g_e2, = ax.plot3D([_rx,_rx+dx], [_ry,_ry+dy], [_rz,_rz+dz], color='green')

        dx,dy,dz = e3[:,i]*frame_scale
        g_e3, = ax.plot3D([_rx,_rx+dx], [_ry,_ry+dy], [_rz,_rz+dz], color='red')
        
        g_Es.append((g_e1, g_e2, g_e3))
       

    set_aspect_equal(ax)


    def animate(k):
        R = Rs[k]
        E = Es[k]
        
        g_R.set_xdata(R[0])
        g_R.set_ydata(R[1])
        g_R.set_3d_properties(R[2])
        
        e1, e2, e3 = E[:,0], E[:,1], E[:,2]

        l = 0
        for i in range(0, R.shape[-1], int(R.shape[-1]/N_frame)):
            g_e1, g_e2, g_e3 = g_Es[l]
            l += 1
            
            _rx, _ry, _rz = R[:,i]
            dx,dy,dz = e1[:,i]*frame_scale
            g_e1.set_xdata([_rx,_rx+dx])
            g_e1.set_ydata([_ry,_ry+dy])
            g_e1.set_3d_properties([_rz,_rz+dz])

            dx,dy,dz = e2[:,i]*frame_scale
            g_e2.set_xdata([_rx,_rx+dx])
            g_e2.set_ydata([_ry,_ry+dy])
            g_e2.set_3d_properties([_rz,_rz+dz])

            dx,dy,dz = e3[:,i]*frame_scale
            g_e3.set_xdata([_rx,_rx+dx])
            g_e3.set_ydata([_ry,_ry+dy])
            g_e3.set_3d_properties([_rz,_rz+dz])
        
        return g_R,

    ani = animation.FuncAnimation(
        fig, animate, interval=1, blit=True, save_count=Rs.shape[0])

    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)
    
    if not save_name is None:
        ani.save(save_name, writer=writer)

    return fig, ani




def vid_scalar(phis, us, fig=None, ax=None, save_name=None):
    if fig is None:
        fig = plt.figure(figsize=(10,10))
    if ax is None:
        ax = plt.axes()

    phi = phis[0]

    g_phi, = ax.plot(us, phi)

    ymin, ymax = np.min(phis), np.max(phis)

    ax.set_ylim(ymin, ymax)
    #ax.set_aspect('equal')

    def animate(i):
        phi = phis[i]
        
        g_phi.set_ydata(phi)

        return g_phi,

    ani = animation.FuncAnimation(
        fig, animate, interval=1, blit=True, save_count=phis.shape[0])

    writer = animation.FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)

    if not save_name is None:
        ani.save(save_name, writer=writer)

    return ani