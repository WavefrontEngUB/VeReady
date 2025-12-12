#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:13:43 2023

@author: artur
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
import cv2
from scipy import special
from scipy import signal
import tabulate as t
#import noise


def definicioAngles(NP):
    
    #skimage.transform.warp_polar
    #https://forum.image.sc/t/polar-transform-and-inverse-transform/40547/3
    
    NPP2 = NP // 2
    
    th = np.linspace(0, np.pi/2, NP)
    lth = np.ones(NP)
    #lth[int(np.round(NP*theta_max/(np.pi/2),0)):NP] = 0
    
    th= np.linspace(0, np.pi/2, NP) * lth
    ph = np.linspace(0, 2 * np.pi, NP)
    
    phiC, thetaC = np.meshgrid(ph, th)
    phiC = phiC.transpose()[::-1,:]
    thetaC = thetaC.transpose()

    phi = cv2.linearPolar(phiC, (NPP2,NPP2), NPP2, cv2.WARP_INVERSE_MAP)
    theta = cv2.linearPolar(thetaC, (NPP2,NPP2), NPP2, cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS)
    
    #exit_pupil = (theta <= theta_max) * ((theta > 0))
    #theta = theta * exit_pupil
    
    #chechPT(phiC, thetaC, phi, theta)
    return phi, theta


def chechPT(phiC, thetaC, phi, theta):
    plt.figure()
    plt.subplot(221)
    plt.imshow(phiC)
    plt.subplot(222)
    plt.imshow(thetaC)
    plt.subplot(223)
    plt.imshow(phi)
    plt.subplot(224)
    plt.imshow(theta)

    
def innerProduct(v1, v2):
    return v1[0]*v2[0] + v1[1]*v2[1] + v1[2]*v2[2] 


def showVector(v):
    figs, axs = plt.subplots(ncols=3)
    axs[0].imshow(np.abs(v[0]))
    axs[1].imshow(np.abs(v[1]))
    axs[2].imshow(np.abs(v[2]))
    
    
def RichardsWolf(Ei, e1, e2, e2p, n1, n2, theta, theta_max, factor, NP):
    Ei = np.array(Ei)
    Einf = (innerProduct(Ei, e1) * e1 + innerProduct(Ei, e2) * e2p) * np.sqrt(n1 / n2) * np.sqrt(np.cos(theta))
    
    #showVector(Einf)
    exit_pupil = (theta <= theta_max) * ((theta > 0))
    exit_pupil[NP//2, NP//2] = 1
    
    Einf = Einf * exit_pupil
    
    Ef = np.array([fft.fftshift(fft.fft2(Einf[0] / np.cos(theta), (factor*NP, factor*NP))),
                   fft.fftshift(fft.fft2(Einf[1] / np.cos(theta), (factor*NP, factor*NP))),
                   fft.fftshift(fft.fft2(Einf[2] / np.cos(theta), (factor*NP, factor*NP)))])
    
    u, v, = np.meshgrid(np.linspace(-1,1,NP), np.linspace(-1,1,NP))
    
    #Ef = Ef * np.exp(1j * 2 * np.pi * 0.5)
    
    center1 = (Ef.shape[1] - Einf.shape[1])//2# + NP//2
    center2 = (Ef.shape[1] + Einf.shape[1])//2# + NP//2#assuming squared arrays
    Efs = np.array([Ef[0, center1:center2, center1:center2],
                    Ef[1, center1:center2, center1:center2],
                    Ef[2, center1:center2, center1:center2]])
    
    
    return Ef, Efs


def invRichardsWolf(Ef, theta, phi, theta_max, factor, NP, n1, n2):
    #Einfs = np.array([fft.ifftshift(fft.ifft2(Ef[0,:,:] * np.cos(theta), (factor*NP, factor*NP))),
    #           fft.ifftshift(fft.ifft2(Ef[1,:,:] * np.cos(theta), (factor*NP, factor*NP))),
    #           fft.ifftshift(fft.ifft2(Ef[2,:,:] * np.cos(theta), (factor*NP, factor*NP)))])
    
    Einfs = np.array([fft.ifftshift(fft.ifft2(Ef[0,:,:], (factor*NP, factor*NP))),
               fft.ifftshift(fft.ifft2(Ef[1,:,:] , (factor*NP, factor*NP))),
               fft.ifftshift(fft.ifft2(Ef[2,:,:] , (factor*NP, factor*NP)))])


    center1 = (Einfs.shape[1] - Ef.shape[1])//2 #+ NP//2 # Remove phase shift. Check!
    center2 = (Einfs.shape[1] + Ef.shape[1])//2 #+ NP//2#assuming squared arrays
    Einf = np.array([Einfs[0, center1:center2, center1:center2],
                    Einfs[1, center1:center2, center1:center2],
                    Einfs[2, center1:center2, center1:center2]])
    

    
    coscut = 0.1
    sqrtcostheta = np.sqrt(np.cos(theta) * (np.cos(theta) >= coscut)  + coscut * (np.cos(theta) < coscut))
    Einf = Einf / np.sqrt(n1 / n2) / sqrtcostheta  
    
    #showVector(Einf)
    
    
    sintheta = np.ma.masked_invalid(np.sin(theta))
    #cosphi = np.ma.masked_invalid(np.cos(phi))
    #sinphi = np.ma.masked_invalid(np.sin(phi))
    
    f2 = Einf[2,:,:] / sintheta
    
    f1a = -Einf[0,:,:] + Einf[2,:,:] * np.cos(phi) * np.cos(theta) / sintheta# / sinphi# / np.sin(phi)
    f1b =  Einf[1,:,:] - Einf[2,:,:] * np.sin(phi) * np.cos(theta) / sintheta # / cosphi #/ np.cos(theta)
    
    
    
    Eix = -f1a  + f2 * np.cos(phi)
    Eiy = f1b  + f2 * np.sin(phi)
    
    #maxim = np.sqrt(np.abs(Eix)**2 + np.abs(Eiy)**2)
    #Eix = np.uint8(255 * np.abs(Eix) / maxim) * np.exp(1j * np.angle(Eix)) 
    #Eiy = np.uint8(255 * np.abs(Eiy) / maxim) * np.exp(1j * np.angle(Eiy)) 
    
 
    Ei = np.array([Eix, Eiy, np.zeros([NP, NP])]) 
    
    
# =============================================================================
#     import matplotlib.pyplot as plt
#     import numpy as np
# 
#     plt.figure()
#     theta = np.linspace(-np.pi/2, np.pi/2, 100)
# 
# 
#     sqrtcostheta = np.sqrt(np.cos(theta) * (np.cos(theta) >= 0.1)  + 0.1 * (np.cos(theta) < 0.1))
#     plt.plot(theta,1/sqrtcostheta)
# =============================================================================
    
    
    return Ei, Einf



    
    
def showFocField(Ei, Efs, NP, zonaVisPp2, fh):
    
    figs, axs = plt.subplots(ncols=4, nrows=4, figsize=(9,11))
    #figs.suptitle(kernelName + ' m = '+str(m))
    
    
    v0 = np.copy(Ei)
    v = np.copy(Efs)
    
    
    
    I0 = np.zeros([3, v0.shape[1], v0.shape[2]])
    I0[0, :, :] = np.abs(v0[0, :, :])**2
    I0[1, :, :] = np.abs(v0[1, :, :])**2
    I0[2, :, :] = I0[0, :, :] + I0[1, :, :]
    I0max = I0[2, :, :].max()
    
    I0[0, :, :] = I0[0, :, :] / I0max
    I0[1, :, :] = I0[1, :, :] / I0max
    I0[2, :, :] = I0[2, :, :] / I0max
    
    I = np.zeros([4, v.shape[1], v.shape[2]])
    I[0, :, :] = np.abs(v[0, :, :])**2
    I[1, :, :] = np.abs(v[1, :, :])**2
    I[2, :, :] = np.abs(v[2, :, :])**2
    I[3, :, :] = I[0, :, :] + I[1, :, :] + I[2, :, :]
    
    Imax = I[3, :, :].max()
    
    I[0, :, :] = I[0, :, :] / Imax
    I[1, :, :] = I[1, :, :] / Imax
    I[2, :, :] = I[2, :, :] / Imax
    I[3, :, :] = I[3, :, :] / Imax
    
    phi_x = np.angle(v[0, :, :])
    phi_y = np.angle(v[1, :, :])
    phi_z = np.angle(v[2, :, :])
    
    
    #nfil, ncol = I[0, :, :].shape
    
    limsup = zonaVisPp2
    liminf = -zonaVisPp2
     
    axs[0, 0].imshow(np.sqrt(I0[0, :, :]), vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    axs[0, 0].set_title('$|E^i_{x}|$')
    axs[0 ,0].grid(True)
    axs[0, 1].imshow(np.sqrt(I0[1, :, :]), vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    axs[0, 1].set_title('$|E^i_{y}|$')
    axs[0 ,1].grid(True)
    axs[0, 2].imshow(np.sqrt(I0[2, :, :]), vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    axs[0, 2].set_title('$\sqrt{ |E^i_{x}|^2+|E^i_{y}|^2}$')
    axs[0, 2].grid(True)
    #axs[0, 3].imshow(pmv0, vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    axs[0, 3].set_title('$\mathbf{E^i}$ polarization map')
    
       
    valmax = np.sqrt(I0max)
    
    nfil, ncol = v0.shape[1], v0.shape[2]
    
    
    axs[0, 3].imshow(np.ones([nfil, ncol]), cmap='gray', vmin=0, vmax=1, extent=((-np.pi/2, np.pi/2, -np.pi/2, np.pi/2)))
    for yy in np.linspace(np.pi/2, -np.pi/2, 20, endpoint=False):
        for xx in np.linspace(-np.pi/2, np.pi/2, 20, endpoint=False):
            jj = int(np.round((xx + np.pi/2)* ncol / (2 * np.pi/2)))
            ii = nfil -int(np.round((yy + np.pi/2)* nfil / (2 * np.pi/2)))
            
            A1 = np.abs(v0[0, ii, jj])
            A2 = np.abs(v0[1, ii, jj])
            delta = np.angle(v0[0, ii, jj])-np.angle(v0[1, ii, jj])
        
            a1 = (A1 * 0.05 / valmax)
            a2 = (A2 * 0.05 / valmax)
            
            t = np.linspace(0, 2*np.pi, 100)
            axs[0, 3].plot(xx + a1*np.cos(t), yy + a2*np.cos(t+delta), 'r')
            
      
    axs[1, 0].imshow(I[0, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    axs[1, 0].set_title('$I_x = |E_x|^2$')
    axs[1, 0].grid()
    axs[1, 1].imshow(I[1, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    axs[1, 1].set_title('$I_y = |E_y|^2$')
    axs[1, 1].grid()
    
    axs[1, 2].imshow(I[0, :, :] + I[1, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    axs[1, 2].set_title('$I_t = |E_x|^2 + |E_y|^2$')  
    axs[1, 2].grid()
       
    axs[1, 3].set_title('$\mathbf{E}$ polarization map')
    axs[1, 3].imshow(np.ones([nfil, ncol]), cmap='gray', vmin=0, vmax=1, extent=((liminf, limsup, liminf, limsup)))
    
    
    valmax = np.sqrt(Imax)
    for yy in np.linspace(limsup, liminf, 20, endpoint=False):
        for xx in np.linspace(liminf, limsup, 20, endpoint=False):
    
            jj = int(np.round((xx + limsup)* ncol / (2 * limsup)))
            ii = nfil - int(np.round((yy + limsup)* ncol / (2 * limsup)))
            #print(np.round(xx,1), np.round(yy,1), ii, jj)
            delta = np.angle(v[0, ii, jj])-np.angle(v[1, ii, jj])
            
            A1 = np.abs(v[0, ii, jj])
            A2 = np.abs(v[1, ii, jj])
        
            a1 = (A1 * 0.1 / valmax)
            a2 = (A2 * 0.1 / valmax)
            
            t = np.linspace(0, 2*np.pi, 100)
            axs[1, 3].plot(xx + a1*np.cos(t), yy + a2*np.cos(t+delta), 'r')
    axs[1,3].set_xlim(liminf, limsup)
    axs[1,3].set_ylim(liminf, limsup)
    #axs[1, 3].axis(liminf, limsup, liminf, limsup)
    axs[1, 3].grid()    
    
    x = np.linspace(liminf, limsup, NP)
    axs[2,0].plot(x, I[0, :, :][NP//2, :], label='$I_x$')
    axs[2,0].plot(x, I[1, :, :][NP//2, :], label='$I_y$')
    axs[2,0].plot(x, I[2, :, :][NP//2, :], label='$I_z$')
    axs[2,0].plot(x, I[1, :, :][NP//2, :] + I[0, :, :][NP//2, :], label='$I_t$')
    axs[2,0].plot(x, I[3, :, :][NP//2, :], label='$I_T$')
    axs[2,0].legend()
    axs[2,0].grid(True)
    
    axs[2, 1].imshow(I[2, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    axs[2, 1].set_title('$I_z = |E_z|^2$')
    #figs.colorbar(eix1, ax=axs[0,2], shrink=1)#, location = 'right')
    axs[2, 1].grid() 
    
    axs[2, 2].imshow(I[3, :, :], vmin = 0, vmax = 1, cmap='gray', extent=(liminf, limsup, liminf, limsup))
    axs[2, 2].set_title('$I_T= |E_x|^2 + |E_y|^2 + |E_z|^2$')
    #axs[2, 2].grid()
    #figs.colorbar(eix2, ax=axs[1,2], shrink=1)#, location = 'right')
    axs[2, 3].axis(False)
    
    axs[3, 0].imshow(phi_x, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
    axs[3, 0].set_title('$\Phi_x$')
    axs[3, 0].grid() 
    axs[3, 1].imshow(phi_y, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
    axs[3, 1].set_title('$\Phi_y$') 
    axs[3, 1].grid() 
    axs[3, 2].imshow(phi_y - phi_x, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
    axs[3, 2].set_title('$\Phi_x - \Phi_y$')
    axs[3, 2].grid() 
    axs[3, 3].imshow(phi_z, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
    axs[3, 3].set_title('$\Phi_z$')
    axs[3, 3].grid() 
    #figs.colorbar(eix3, ax=axs[2,2], shrink=1)#, location = 'right')
    
    figs.tight_layout()
    figs.show()
    
    return figs



def showFocField2(Ei, Efs, IRM, Egs, NP, zonaVisPp2):
    
    #figs, axs = plt.subplots(ncols=4, nrows=3, figsize=(9,9))
    figs, axs = plt.subplots(ncols=5, nrows=2, figsize=(16, 9))
    
    #figs.suptitle(kernelName + ' m = '+str(m))
    
    
    v0 = np.copy(Ei)
    v = np.copy(Efs)
    
    
    
    I0 = np.zeros([3, v0.shape[1], v0.shape[2]])
    I0[0, :, :] = np.abs(v0[0, :, :])**2
    I0[1, :, :] = np.abs(v0[1, :, :])**2
    I0[2, :, :] = I0[0, :, :] + I0[1, :, :]
    I0max = I0[2, :, :].max()
    
    I0[0, :, :] = I0[0, :, :] / I0max
    I0[1, :, :] = I0[1, :, :] / I0max
    I0[2, :, :] = I0[2, :, :] / I0max
    
    I = np.zeros([4, v.shape[1], v.shape[2]])
    I[0, :, :] = np.abs(v[0, :, :])**2
    I[1, :, :] = np.abs(v[1, :, :])**2
    I[2, :, :] = np.abs(v[2, :, :])**2
    I[3, :, :] = I[0, :, :] + I[1, :, :] + I[2, :, :]
    
    Imax = I[3, :, :].max()
    
    I[0, :, :] = I[0, :, :] / Imax
    I[1, :, :] = I[1, :, :] / Imax
    I[2, :, :] = I[2, :, :] / Imax
    I[3, :, :] = I[3, :, :] / Imax
    
    
    phi_x = np.angle(v[0, :, :])
    phi_y = np.angle(v[1, :, :])
    phi_z = np.angle(v[2, :, :])
    
    
    #nfil, ncol = I[0, :, :].shape
    
    limsup = zonaVisPp2
    liminf = -zonaVisPp2
     
    axs[0, 0].imshow(I0[0, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    axs[0, 0].set_title('$I_{x}^i = |E^i_{x}|^2$')
    axs[0 ,0].grid(True)
    axs[0, 1].imshow(I0[1, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    axs[0, 1].set_title('$I_{y}^i = |E^i_{y}|^2$')
    axs[0 ,1].grid(True)
    axs[0, 2].imshow(I0[2, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    axs[0, 2].set_title('$I_{t}^i = |E^i_{x}|^2+|E^i_{y}|^2$')
    axs[0, 2].grid(True)
    #axs[0, 3].imshow(pmv0, vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    axs[0, 3].set_title('$\mathbf{E^i}$ polarization map')
    
       
    valmax = np.sqrt(I0max)
    
    nfil, ncol = v0.shape[1], v0.shape[2]
    
    
    axs[0, 3].imshow(np.ones([nfil, ncol]), cmap='gray', vmin=0, vmax=1, extent=((-np.pi/2, np.pi/2, -np.pi/2, np.pi/2)))
    for yy in np.linspace(np.pi/2, -np.pi/2, 20, endpoint=False):
        for xx in np.linspace(-np.pi/2, np.pi/2, 20, endpoint=False):
            jj = int(np.round((xx + np.pi/2)* ncol / (2 * np.pi/2)))
            ii = nfil -int(np.round((yy + np.pi/2)* nfil / (2 * np.pi/2)))
            
            A1 = np.abs(v0[0, ii, jj])
            A2 = np.abs(v0[1, ii, jj])
            delta = np.angle(v0[0, ii, jj])-np.angle(v0[1, ii, jj])
        
            a1 = (A1 * 0.2 / valmax)
            a2 = (A2 * 0.2 / valmax)
            
            t = np.linspace(0, 2*np.pi, 100)
            axs[0, 3].plot(xx + a1*np.cos(t), yy + a2*np.cos(t+delta), 'r')
            
      
    axs[1, 0].imshow(I[0, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    axs[1, 0].set_title('$I_x = |E_x|^2$')
    axs[1, 0].grid()
    axs[1, 1].imshow(I[1, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    axs[1, 1].set_title('$I_y = |E_y|^2$')
    axs[1, 1].grid()
    
    #axs[1, 2].imshow(I[0, :, :] + I[1, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    #axs[1, 2].set_title('$I_t = |E_x|^2 + |E_y|^2$')  
    #axs[1, 2].grid()
       
    axs[1, 3].set_title('$\mathbf{E}$ polarization map')
    axs[1, 3].imshow(np.ones([nfil, ncol]), cmap='gray', vmin=0, vmax=1, extent=((liminf, limsup, liminf, limsup)))
    
    
    valmax = np.sqrt(Imax)
    for yy in np.linspace(limsup, liminf, 20, endpoint=False):
        for xx in np.linspace(liminf, limsup, 20, endpoint=False):
    
            jj = int(np.round((xx + limsup)* ncol / (2 * limsup)))
            ii = nfil - int(np.round((yy + limsup)* ncol / (2 * limsup)))
            #print(np.round(xx,1), np.round(yy,1), ii, jj)
            delta = np.angle(v[0, ii, jj])-np.angle(v[1, ii, jj])
            
            A1 = np.abs(v[0, ii, jj])
            A2 = np.abs(v[1, ii, jj])
        
            a1 = (A1 * 0.1 / valmax)
            a2 = (A2 * 0.1 / valmax)
            
            t = np.linspace(0, 2*np.pi, 100)
            axs[1, 3].plot(xx + a1*np.cos(t), yy + a2*np.cos(t+delta), 'r')
    axs[1,3].set_xlim(liminf, limsup)
    axs[1,3].set_ylim(liminf, limsup)
    #axs[1, 3].axis(liminf, limsup, liminf, limsup)
    axs[1, 3].grid()    
    
    x = np.linspace(liminf, limsup, NP)
    #axs[2,0].plot(x, I[0, :, :][NP//2, :], label='$I_x$')
    #axs[2,0].plot(x, I[1, :, :][NP//2, :], label='$I_y$')
    #axs[2,0].plot(x, I[1, :, :][NP//2, :], label='$I_y$')
    #axs[2,0].plot(x, I[2, :, :][NP//2, :], label='$I_z$')
    #axs[2,0].plot(x, I[1, :, :][NP//2, :] + I[0, :, :][NP//2, :], label='$I_t$')
    axs[0,4].plot(x, I[3, :, :][NP//2, :] / I[3, :, :][NP//2, :].max(), label='$I_C$')
   
    axs[0,4].grid(True)
    
    axs[1, 2].imshow(I[2, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    axs[1, 2].set_title('$I_z = |E_z|^2$')
    #figs.colorbar(eix1, ax=axs[0,2], shrink=1)#, location = 'right')
    axs[1, 2].grid() 
    
    axs[1, 4].imshow(I[3, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    axs[1, 4].set_title('$I_C= |E_x|^2 + |E_y|^2 + |E_z|^2$')
    axs[1, 4].grid() 
    #figs.colorbar(eix2, ax=axs[1,2], shrink=1)#, location = 'right')
    
    #Itarget = np.abs(Etarget[0,:,:])**2 + np.abs(Etarget[1,:,:])**2 + np.abs(Etarget[2,:,:])**2
    IRM = IRM / IRM[NP//2,:].max()
    Igs = np.abs(Egs[0,:,:])**2 + np.abs(Egs[1,:,:])**2 + np.abs(Egs[2,:,:])**2
    Igs = Igs / Igs.max()

    #axs[2, 1].imshow(IRM, vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    #axs[2, 1].set_title('$I_{\mathrm{Target}}$')
    #axs[2, 1].grid() 
    #axs[2, 2].imshow(Igs, vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    #axs[2, 2].set_title('$I_{GS}$ (Gold standard)')
    #axs[2, 2].grid() 
    
    axs[0, 4].plot(x, Igs[NP//2, :], label='$I_{GS}$')
    axs[0, 4].plot(x, IRM[NP//2, :], label='$I_{\mathrm{Target}}$')
    axs[0, 4].legend()
    #axs[2, 3].axis(False)
    
# =============================================================================
#     axs[3, 0].imshow(phi_x, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
#     axs[3, 0].set_title('$\Phi_x$')
#     axs[3, 0].grid() 
#     axs[3, 1].imshow(phi_y, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
#     axs[3, 1].set_title('$\Phi_y$') 
#     axs[3, 1].grid() 
#     axs[3, 2].imshow(phi_y - phi_x, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
#     axs[3, 2].set_title('$\Phi_x - \Phi_y$')
#     axs[3, 2].grid() 
#     axs[3, 3].imshow(phi_z, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
#     axs[3, 3].set_title('$\Phi_z$')
#     axs[3, 3].grid() 
#     #figs.colorbar(eix3, ax=axs[2,2], shrink=1)#, location = 'right')
# =============================================================================
    
    figs.tight_layout()
    figs.show()
    
    return figs




def showFocField3(Ei, Efs, IRM, Egs, NP, zonaVisPp2):
    #import matplotlib as mpl
    #mpl.rcParams['font.size'] = 13
    figs = plt.figure(figsize=(10, 10))
    
    
    v0 = np.copy(Ei)
    v = np.copy(Efs)
    
    
    
    I0 = np.zeros([3, v0.shape[1], v0.shape[2]])
    I0[0, :, :] = np.abs(v0[0, :, :])**2
    I0[1, :, :] = np.abs(v0[1, :, :])**2
    I0[2, :, :] = I0[0, :, :] + I0[1, :, :]
    I0max = I0[2, :, :].max()
    
    I0[0, :, :] = I0[0, :, :] / I0max
    I0[1, :, :] = I0[1, :, :] / I0max
    I0[2, :, :] = I0[2, :, :] / I0max
    
    I = np.zeros([4, v.shape[1], v.shape[2]])
    I[0, :, :] = np.abs(v[0, :, :])**2
    I[1, :, :] = np.abs(v[1, :, :])**2
    I[2, :, :] = np.abs(v[2, :, :])**2
    I[3, :, :] = I[0, :, :] + I[1, :, :] + I[2, :, :]
    
    Imax = I[3, :, :].max()
    
    I[0, :, :] = I[0, :, :] / Imax
    I[1, :, :] = I[1, :, :] / Imax
    I[2, :, :] = I[2, :, :] / Imax
    I[3, :, :] = I[3, :, :] / Imax
    
    
    phi_x = np.angle(v[0, :, :])
    phi_y = np.angle(v[1, :, :])
    phi_z = np.angle(v[2, :, :])
    
    
        
    limsup = zonaVisPp2
    liminf = -zonaVisPp2
     
    plt.subplot(441)
    plt.imshow(np.sqrt(I0[0, :, :]), vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    plt.title('$|E_{0x}|$')
    plt.grid(True)
    
    plt.subplot(442)
    plt.imshow(np.sqrt(I0[1, :, :]), vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    plt.title('$ |E_{0y}|$')
    plt.grid(True)
    
    plt.subplot(443)
    plt.imshow(np.sqrt(I0[2, :, :]), vmin = 0, vmax = 1, cmap='jet', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    plt.title('$\sqrt{|E_{0x}|^2+|E_{0y}|^2}$')
    plt.grid(True)
    
    plt.subplot(444)
    plt.title('$\mathbf{E_0}$ polarization map')
    
       
    valmax = np.sqrt(I0max)
    
    nfil, ncol = v0.shape[1], v0.shape[2]
    
    plt.imshow(np.ones([nfil, ncol]), cmap='gray', vmin=0, vmax=1, extent=((-np.pi/2, np.pi/2, -np.pi/2, np.pi/2)))
    for yy in np.linspace(np.pi/2, -np.pi/2, 20, endpoint=False):
        for xx in np.linspace(-np.pi/2, np.pi/2, 20, endpoint=False):
            jj = int(np.round((xx + np.pi/2)* ncol / (2 * np.pi/2)))
            ii = nfil -int(np.round((yy + np.pi/2)* nfil / (2 * np.pi/2)))
            
            A1 = np.abs(v0[0, ii, jj])
            A2 = np.abs(v0[1, ii, jj])
            delta = np.angle(v0[0, ii, jj])-np.angle(v0[1, ii, jj])
        
            a1 = (A1 * 0.2 / valmax)
            a2 = (A2 * 0.2 / valmax)
            
            t = np.linspace(0, 2*np.pi, 100)
            plt.plot(xx + a1*np.cos(t), yy + a2*np.cos(t+delta), 'r')
            
    
    plt.subplot(445)
    plt.imshow(I[0, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    plt.title('$I_x = |E_x|^2$')
    plt.grid()
    
    
    plt.subplot(446)
    plt.imshow(I[1, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    plt.title('$I_y = |E_y|^2$')
    plt.grid()
    

    plt.subplot(448)   
    plt.title('$\mathbf{E}$ polarization map')
    plt.imshow(np.ones([nfil, ncol]), cmap='gray', vmin=0, vmax=1, extent=((liminf, limsup, liminf, limsup)))
    
    
    valmax = np.sqrt(Imax)
    for yy in np.linspace(limsup, liminf, 20, endpoint=False):
        for xx in np.linspace(liminf, limsup, 20, endpoint=False):
    
            jj = int(np.round((xx + limsup)* ncol / (2 * limsup)))
            ii = nfil - int(np.round((yy + limsup)* ncol / (2 * limsup)))
            #print(np.round(xx,1), np.round(yy,1), ii, jj)
            delta = np.angle(v[0, ii, jj])-np.angle(v[1, ii, jj])
            
            A1 = np.abs(v[0, ii, jj])
            A2 = np.abs(v[1, ii, jj])
        
            a1 = (A1 * 0.1 / valmax)
            a2 = (A2 * 0.1 / valmax)
            
            t = np.linspace(0, 2*np.pi, 100)
            plt.plot(xx + a1*np.cos(t), yy + a2*np.cos(t+delta), 'r')
    plt.xlim(liminf, limsup)
    plt.ylim(liminf, limsup)
    plt.grid()    
    
    x = np.linspace(liminf, limsup, NP)

    plt.subplot(447)
    plt.imshow(I[2, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    plt.title('$I_z = |E_z|^2$')
    plt.grid() 
    
    plt.subplot(223)
    plt.imshow(I[3, :, :], vmin = 0, vmax = 1, cmap='jet', extent=(liminf, limsup, liminf, limsup))
    plt.title('$I_C= |E_x|^2 + |E_y|^2 + |E_z|^2$')
    plt.grid() 

    IRM = IRM / IRM[NP//2,:].max()
    Igs = np.abs(Egs[0,:,:])**2 + np.abs(Egs[1,:,:])**2 + np.abs(Egs[2,:,:])**2
    Igs = Igs / Igs.max()



    plt.subplot(224)    
    plt.plot(x, I[3, :, :][NP//2, :] / I[3, :, :][NP//2, :].max(), label='$I_C$')      
    plt.plot(x, Igs[NP//2, :], label='$I_{GS}$')
    plt.plot(x, IRM[NP//2, :], label='$I_{\mathrm{Target}}$')
    plt.grid(True)
    plt.legend()  
    plt.subplots_adjust(top=0.946,bottom=0.047,left=0.083,right=0.975,hspace=0.1,wspace=0.372)
    # plt.tight_layout()
    # plt.show()
    
    return figs



def aberra(theta, phi, Z5, Z8, Z9, Z12):
    rho = np.sin(theta)
    astig = Z5 * rho**2 * np.cos(2*phi)
    comax = Z8 * (3*rho**3 - 2*rho) * np.cos(phi)
    trefo = Z9 * rho**3 * np.cos(3 * phi)
    esfer = Z12 * (6*rho - 4*rho**2 +1)
    wa = np.exp(2j * np.pi * (astig + comax + trefo + esfer))
    return wa

def kernelDef(kernelName, theta, phi, m, NP, f_0, theta_max, fileinput):#, Z5, Z8, Z9, Z12):
    kernel = np.exp(-np.sin(theta)**2 / f_0**2 / np.sin(theta_max)**2) #* aberra(theta, phi, Z5, Z8, Z9, Z12)# * apertura#* morph.disk(NP//2-0.5)

    if kernelName == 'Gaussian linear':
        kernel = kernel *  np.exp(1j * m * phi)
        Ei = [kernel, np.zeros([NP, NP]), np.zeros([NP, NP])]
    if kernelName == 'Gaussian circular':
        kernel =  np.sin(theta) * kernel *  np.exp(1j * m * phi) 
        Ei = [kernel, 1j*kernel, np.zeros([NP, NP])]
    if kernelName == 'Gaussian radial': 
        kernel = np.sin(theta) * kernel * np.exp(1j * m * phi)
        Ei = [kernel * np.cos(phi), kernel * np.sin(phi), np.zeros([NP, NP])]
    if kernelName == 'Gaussian azimuthal': 
        kernel = np.sin(theta) * kernel * np.exp(1j * m *phi)
        Ei = [-kernel * np.sin(phi), kernel * np.cos(phi), np.zeros([NP, NP])]
# =============================================================================
#     if kernelName == 'Laguerre 01 linear':
#         kernel = kernel *  np.exp(1j * m * phi) * np.sin(theta)
#         Ei = [kernel, np.zeros([NP, NP]), np.zeros([NP, NP])]
#     if kernelName == 'Laguerre 01 circular':
#         kernel = kernel *  np.exp(1j * m * phi) * np.sin(theta)
#         Ei = [kernel, -1j*kernel, np.zeros([NP, NP])]
#     if kernelName == 'Laguerre 01 radial':
#         kernel = kernel *  np.exp(1j * m * phi) * np.sin(theta)
#         Ei = [kernel * np.cos(phi), kernel * np.sin(phi), np.zeros([NP, NP])]
#     if kernelName == 'Laguerre 01 azimuthal':
#         kernel = kernel *  np.exp(1j * m * phi) * np.sin(theta)
#         Ei = [-kernel * np.sin(phi), kernel * np.cos(phi), np.zeros([NP, NP])]
#     if kernelName == 'Needle azimuthal':
#         kernel = kernel * np.sinc(2*N*(np.cos(theta)-0.5*(1+np.cos(theta_max)))
#                                   /(1-np.cos(theta_max))) * np.sin(theta)
#         Ei = [kernel * np.exp(1j * m * phi) * np.sin(phi), -kernel * np.cos(phi), np.zeros([NP, NP])]
#     if kernelName =='Bessel J1 linear':
#         N=20
#         kernel = kernel * np.exp(1j * m * phi) * special.jv(1, N * np.sin(theta) / f_0 / np.sin(theta_max)) 
#         Ei = [kernel, np.zeros([NP, NP]), np.zeros([NP, NP])]
#     if kernelName =='Bessel J1 circular':
#         N=20
#         kernel = kernel * np.exp(1j * m * phi) * special.jv(1, N * np.sin(theta) / f_0 / np.sin(theta_max)) 
#         Ei = [kernel, kernel * 1j, np.zeros([NP, NP])]
#     if kernelName == 'Bessel J1 radial': 
#         N=20
#         kernel = kernel * np.exp(1j * m * phi) * special.jv(1, N * np.sin(theta) / f_0 / np.sin(theta_max))
#         Ei = [kernel * np.cos(phi), kernel * np.sin(phi), np.zeros([NP, NP])]
#     if kernelName == 'Bessel J1 azimuthal': 
#         N=20
#         kernel = kernel * np.exp(1j * m * phi) * special.jv(1, N * np.sin(theta) / f_0 / np.sin(theta_max))
#         Ei = [kernel * np.sin(phi), -kernel * np.cos(phi), np.zeros([NP, NP])] 
# =============================================================================
    if kernelName == 'File':
        Ei = np.load(fileinput)
        Ei[2,:,:] = np.zeros([NP, NP])
    
    exit_pupil = (theta <= theta_max) * ((theta > 0))
    exit_pupil[NP//2, NP//2] = 1
    Ei = Ei * np.array([exit_pupil, exit_pupil, exit_pupil])   
    return Ei
        
    
def showInputField(Ef, Einf, Ei):
    
    Imax = np.max(np.abs(Ef[0,:,:])**2 +  np.abs(Ef[1,:,:])**2 +  np.abs(Ef[2,:,:])**2)

    Ix = np.abs(Ef[0,:,:])**2 / Imax
    Iy = np.abs(Ef[1,:,:])**2 / Imax
    Iz = np.abs(Ef[2,:,:])**2 / Imax
    
    plt.figure(figsize=(9,11))
    plt.subplot(331)
    plt.imshow(Ix, vmin=0, vmax=1, cmap='jet')
    plt.title('$I_{fx}$')
    plt.subplot(332)
    plt.imshow(Iy, vmin=0, vmax=1, cmap='jet')
    plt.title('$I_{fy}$')
    plt.subplot(333)
    plt.imshow(Iz, vmin=0, vmax=1, cmap='jet')
    plt.title('$I_{fz}$')
    plt.subplot(334)
    plt.imshow(np.abs(Einf[0,:,:]))
    plt.title('$E_{\infty x}$')
    plt.subplot(335)
    plt.imshow(np.abs(Einf[1,:,:]))
    plt.title('$E_{\infty y}$')
    plt.subplot(336)
    plt.imshow(np.abs(Einf[2,:,:]))
    plt.title('$E_{\infty z}$')
    plt.subplot(337)
    plt.imshow(np.abs(Ei[0, :, :])**2, cmap='jet')
    plt.title('$E_x$')
    plt.subplot(338)
    plt.imshow(np.abs(Ei[1, :, :])**2, cmap='jet')
    plt.title('$E_y$')
    plt.subplot(339)
    plt.plot(np.abs(Ei[0, :, 64]))
    plt.plot(np.abs(Ei[1, :, 64]))
    
    return

def phaseShiftEstimation(NP, factor):
    
    supmat = np.zeros([int(NP), int(NP)])
    supmat[NP//2, NP//2] = NP*NP 
    supmattf = fft.fftshift(fft.fft2(supmat, (int(factor*NP), int(factor*NP))))
    
    center1 = (int(factor*NP) - int(NP))//2
    center2 = (int(factor*NP) + int(NP))//2
    phase = np.exp(1j * np.angle(supmattf[center1:center2, center1:center2]))
    
    return np.array([phase, phase, phase])


def fwhm(Efs, NP, zonaVisPp2):
    Ifprof = np.abs(Efs[0, NP//2, :])**2 + np.abs(Efs[1, NP//2, :])**2 + \
             np.abs(Efs[2, NP//2, :])**2
    Ifprofmax = Ifprof.max()
    Ifprof = Ifprof / Ifprofmax         
    
    r = np.linspace(-zonaVisPp2, zonaVisPp2, NP)
    
    maxs = signal.argrelmax(Ifprof * (Ifprof > 0.95))[0]
   
    if maxs.size == 2:
        peak2peak = r[maxs[1]] - r[maxs[0]]
        halfmaximum = signal.argrelmin(np.abs(Ifprof-0.5))[0]
        fwhm = r[halfmaximum[2]] - r[halfmaximum[1]]
        #print()
        #print(t.tabulate([['Peak to peak = ', peak2peak],['Ring FWHM = ', fwhm]] ))
    if maxs.size == 1:
        fwhm = 2 * np.abs(r[np.argmax((Ifprof >= 0.5))])
        #print()
        #print(t.tabulate([['Single peak maximum',],['FWHM = ', fwhm]]))
    if maxs.size > 2 or maxs.size==0:
        fwhm = 10.0
    return fwhm, Ifprof, maxs.size



def filehandling(datalist, Ifprof, figs, comment):
          
    try:
        numcase = np.loadtxt('case.txt')
    except:
        numcase = -1
    numcase = int(numcase + 1)
    
    np.savetxt('case.txt',np.array([numcase], dtype=int))
    f = open("readme.txt", "a")
    f.write(str(np.array([numcase])) +':   '+ comment)
    f.close()
    
    
    filenameprofile = str(numcase).zfill(4)+'____profile.npy'
    filenamedata    = str(numcase).zfill(4)+'_______data.npy'  
    filenamepng =     str(numcase).zfill(4)+'_irradiance.png'       
    np.save(filenameprofile, Ifprof)
    np.save(filenamedata, datalist )
    figs.savefig(filenamepng)
    
    return



# =============================================================================
# def perlin(NP):
#     octaves = 1
#     freq = 32.0 * np.random.rand()  * octaves
# 
#     noiseim = np.zeros([NP, NP])
#     for ii in range(NP):
#         for jj in range(NP):
#             value= int(noise.pnoise2(ii / freq, jj / freq, octaves=1)* 127.0 + 128.0)
#             noiseim[ii, jj] = value
#     return noiseim
# =============================================================================




def Stokes(Efs):
    S0 = np.abs(Efs[0])**2 + np.abs(Efs[1])**2 
    S1 = np.abs(Efs[0])**2 - np.abs(Efs[1])**2
    S2 = 2 * (Efs[0]*np.conj(Efs[1])).real
    S3 = -2 * (Efs[0]*np.conj(Efs[1])).imag
    
    #S1 = np.ma.masked_invalid(S1 / S0)
    #S2 = np.ma.masked_invalid(S2 / S0)
    #S3 = np.ma.masked_invalid(S3 / S0)
    
        
    StPar = np.concatenate([S0, S1, S2, S3, np.abs(Efs[2])**2 ], axis=1)
    return StPar
    
    
def StokesInt(Efs):
    #S0 = np.abs(Efs[0])**2 + np.abs(Efs[1])**2 
    #S1 = np.abs(Efs[0])**2 - np.abs(Efs[1])**2
    #S2 = 2 * (Efs[0]*np.conj(Efs[1])).real
    #S3 = -2 * (Efs[0]*np.conj(Efs[1])).imag
    
    #S1 = np.ma.masked_invalid(S1 / S0)
    #S2 = np.ma.masked_invalid(S2 / S0)
    #S3 = np.ma.masked_invalid(S3 / S0)
    
    I0 = np.abs(Efs[0])**2
    I45 = 0.5 * np.abs(Efs[0] + Efs[1])**2 
    I90 = np.abs(Efs[1])**2 
    I135 = 0.5 * np.abs(Efs[0] - Efs[1])**2 
    I45C = 0.5 * np.abs(Efs[0] - 1j * Efs[1])**2 
    I135C = 0.5 * np.abs(Efs[0] + 1j * Efs[1])**2
    Iz = np.abs(Efs[2])**2
    
    StInt_image = np.concatenate([I0, I45, I90, I135, I45C, I135C, Iz], axis=1)
    StInt_array = np.zeros([Efs.shape[1], Efs.shape[2], 7])

    StInt_array[:,:,0] = I0
    StInt_array[:,:,1] = I45
    StInt_array[:,:,2] = I90
    StInt_array[:,:,3] = I135
    StInt_array[:,:,4] = I45C
    StInt_array[:,:,5] = I135C
    StInt_array[:,:,6] = Iz

    return StInt_image, StInt_array    
    

# =============================================================================
# def zernikeRadial_nm(n, m, NP):
#     rho = np.linspace(-1, 1 , NP)
#     R=np.zeros(NP)
#     for l in np.arange(0, 0.5*(n-m)+1):
#         factor = (-1)**l * special.factorial(n-l) / special.factorial(l) / special.factorial(0.5*(n+m) -l) / special.factorial(0.5*(n-m) -l)
#         potencia = n - 2 * l
#         R += factor * np.abs(rho)**potencia
#     poli = np.round(np.polyfit(np.abs(rho), R, n),0)
#     #print(n, m, poli)
#     return poli
# =============================================================================

def zernikeRadial_nm(n, m):
    #rho = np.linspace(-1, 1 , NP)
    #R=np.zeros(NP)
    poli = np.zeros(n+1)
    for l in np.arange(0, 0.5*(n-m)+1):
        factor = (-1)**l * special.factorial(n-l) / special.factorial(l) / special.factorial(0.5*(n+m) -l) / special.factorial(0.5*(n-m) -l)
        potencia = n - 2 * l
        #print(potencia, factor)
        poli[int(potencia)] = factor
        #R += factor * np.abs(rho)**potencia
    return poli

def Zernike2D(n,m, phi, NP):
        poli = zernikeRadial_nm(n, np.abs(m), NP)
        X, Y = np.meshgrid(np.linspace(-1,1,NP), np.linspace(-1,1,NP))
        Rho = np.sqrt(X*X + Y*Y)
        if m >= 0:
            Z = np.polyval(poli, Rho) * np.cos(m * phi) * (Rho<=1)
        else:
            Z = np.polyval(poli, Rho) * np.sin(m * phi) * (Rho<=1)
        return Z
    
def showPhase(Ei, Efs, zonaVisPp2):
    
    limsup = zonaVisPp2
    liminf = -zonaVisPp2
    
    

    plt.figure(figsize=(10, 6))
    
    
    Eic = np.copy(Ei)
    Efc = np.copy(Efs)
    
    phase_ix = np.angle(Eic[0,:,:])
    phase_iy = np.angle(Eic[1,:,:])
    phase_fx = np.angle(Efc[0,:,:])
    phase_fy = np.angle(Efc[1,:,:])    

    phase_fx = (phase_fx < -3.1) * np.pi + (phase_fx >= -3.1) * phase_fx 
    phase_fy = (phase_fy < -3.1) * np.pi + (phase_fy >= -3.1) * phase_fy 
    
    phase_fx = phase_fx * (np.abs(Efc[0,:,:])>0.01)
    phase_fy = phase_fy * (np.abs(Efc[1,:,:])>0.01)

    plt.subplot(221)
    plt.imshow(phase_ix, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    plt.title('$\Phi_{0x}$')
    plt.grid() 
    
    plt.subplot(222) 
    plt.imshow(phase_iy, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    plt.title('$\Phi_{0y}$') 
    plt.grid() 
    
    
    
    # plt.subplot(233)
    # plt.imshow(phase_iy - phase_ix, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(-np.pi/2, np.pi/2, -np.pi/2, np.pi/2))
    # plt.title('$\Phi_{0y} - \Phi_{0x}$')
    # plt.grid() 
    # plt.colorbar(location = 'right')
    
    
    plt.subplot(223)
    plt.imshow(phase_fx, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
    plt.title('$\Phi_x$')
    plt.grid() 
    
    plt.subplot(224) 
    plt.imshow( phase_fy, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
    plt.title('$\Phi_y$') 
    plt.grid() 
    #plt.colorbar(location = 'bottom')
    
    # plt.subplot(236)
    # plt.imshow(phase_fy - phase_fx, vmin = -np.pi, vmax = np.pi, cmap='seismic', extent=(liminf, limsup, liminf, limsup))
    # plt.title('$\Phi_y - \Phi_x$')
    # plt.grid() 

    #plt.colorbar(location = 'bottom')
    
    plt.tight_layout()
    plt.show()
    
    return True

def normalizeEi(Ei):
    
    #Eia = np.abs(Ei)
    #Eia = np.float64(np.uint8(255 * Eia / Eia.max()) / 255)
    
    #Eip = np.deg2rad((360 / 255) * np.uint8(255 * np.angle(Ei, deg=True) / 360))
    
        
    #return Eia * np.exp(1j * Eip)
    
    maxim = np.abs(Ei).max()
    
    return np.round(128 * np.real(Ei) / maxim) + 1j *  np.round(128 * np.imag(Ei) / maxim)