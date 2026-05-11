#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 10:51:57 2025

@author: artur
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy.optimize import dual_annealing, minimize, differential_evolution
from scipy import integrate, special
import cv2
import time
from tqdm import tqdm
import sys


plt.style.use('default')
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14

#%%

def costFunc(alphas):
    """
    
    Parameters
    ----------
    alphas : Alpha coeefficnets according to Eq. (8)

    Returns
    -------
    fC : Cost Function - minimum distance-  for SA

    """
    fC = np.abs((inten(alphas)- target)).sum()
    print(fC) # it can be commented. But provides insight about the convergence of the SA
    return  fC


def inten(alphas):
    """
    Parameters
    ----------
    alphas : TYPE
        DESCRIPTION.

    Returns
    -------
    Intensity according to Eq. (8)

    """
    summand = np.zeros(NPP2, dtype=complex)
    for n in range(0, alphas.size):
        if n%2 == 0:
            alphas[n]=0
        summand += alphas[n] * a**2 * (-1)**((n-1)/2) * np.sqrt(1) *\
                     special.jv(n+1, 2*np.pi*n2*r*a) / (2*np.pi*n2*r*a) # np.sqrt(n+1)
    res = np.abs(summand)**2
    return res / res.max()


def cin(x, r0):
    """
    
    Parameters
    ----------
    x : x in Eq. (6)
    r0 : independent variable r in Eq. (6)

    Returns
    -------
    epsilon_2 : Charo's kernel

    """
    epsilon_2 = 0
    for n in range(1, coeficients.size, 2):       
        coefZR = coeffZernikeRadial_nm(n, 1)[::-1]
        epsilon_2 += coeficients[n] * np.polyval(coefZR, x) * special.jv(1, 2*np.pi*n2*r0*a*x) * x 
    return epsilon_2  #np.exp(-x**2 / f_0**2 / a**2)  

def cinzr(x, r0, z):
    """

    Parameters
    ----------
    x : x in Eq. (6)
    r0 : independent variable r in Eq. (6)
    z: propagation distance in lambdas

    Returns
    -------
    az : Charo's kernel * cos(k *z * x),
         Charo's kernel * sin(k *z * x),
    """
    #return cin(x, r0) * np.cos(2 * np.pi * z * x), cin(x, r0) * np.sin(2 * np.pi * z * x) 
   
    return cin(x, r0) * np.cos(2 * np.pi * n2* z * np.sqrt(1-(a*x)**2)) 

def cinzi(x, r0, z):
    """

    Parameters
    ----------
    x : x in Eq. (6)
    r0 : independent variable r in Eq. (6)
    z: propagation distance in lambdas

    Returns
    -------
    az : Charo's kernel * cos(k *z * x),
         Charo's kernel * sin(k *z * x),
    """
    #return cin(x, r0) * np.cos(2 * np.pi * z * x), cin(x, r0) * np.sin(2 * np.pi * z * x) 
   
    return cin(x, r0) * np.sin(2 * np.pi * n2* z * np.sqrt(1-(a*x)**2)) 


def gaz(x, r0): 
    """

    Parameters
    ----------
    x : x in Eq. (6)
    r0 : independent variable r in Eq. (6)

    Returns
    -------
    az : Gaussian Azimuthal kernel

    """
    az = x * np.exp(-x**2 / f_0**2 / a**2) * special.jv(1, 2*np.pi*n2*r0*x) * x / (1 - (a*x)**2)**0.25
    return az



def gazzr(x, r0, z):
    """

    Parameters
    ----------
    x : x in Eq. (6)
    r0 : independent variable r in Eq. (6)
    z: propagation distance in lambdas

    Returns
    -------
    az : Gaussian Azimuthal kernel * cos(k *z * x),
         Gaussian Azimuthal kernel * sin(k *z * x),

    """
    #return gaz(x, r0) * np.cos(2 * np.pi * z * x), gaz(x, r0) * np.sin(2 * np.pi * z * x) 
        
    return gaz(x, r0) * np.cos(2 * np.pi * n2* z * (1 - (a*x)**2)**0.5) 
           

def gazzi(x, r0, z):
    """

    Parameters
    ----------
    x : x in Eq. (6)
    r0 : independent variable r in Eq. (6)
    z: propagation distance in lambdas

    Returns
    -------
    az : Gaussian Azimuthal kernel * cos(k *z * x),
         Gaussian Azimuthal kernel * sin(k *z * x),

    """
    #return gaz(x, r0) * np.cos(2 * np.pi * z * x), gaz(x, r0) * np.sin(2 * np.pi * z * x) 
        
    return gaz(x, r0) * np.cos(2 * np.pi * n2* z * (1 - (a*x)**2)**0.5) 


def coeffZernikeRadial_nm(n, m): 
    """
    Parameters
    ----------
    n : Radial Zernike Parameter n
    m : Radial Zernike Parameter m. In the present problem, m=1

    Returns
    -------
    poli : Radial Zernike Polynomial Coefficient

    """
    poli = np.zeros(n+1)
    for l in np.arange(0, 0.5*(n-m)+1):
        factor = (-1)**l * special.factorial(n-l) / (special.factorial(l) * special.factorial((n+m)//2 -l) * special.factorial((n-m)//2 -l))
        potencia = n - 2 * l
        poli[np.int32(potencia)]=round(factor,0)
    return poli

#%%
"""
Step 1: parameter definition
"""
plt.close('all')
plt.interactive(False)
toc = time.time()

print("Step 1: parameter definition")
NP = 2048      # Number of samples 
NPP2 = NP // 2    # Half number of samples
NPP4 = NP // 4
NPP8 = NP // 8
zonaVisPp2 = 6 #8    # Radius of the Visualization (in \lambdas) 
nterms = 8  # number of Zernike polynomials terms used (degree; 0, 2, 4,...) 
coef = 6        # J1 function scale factor; coef=5: argmax equivalent to GS, coef=6: superesolution, coef=7, does not converge
a = 0.9373          # a = sin(theta_{max})^2 / n_2 (NA in air)
n2 = 1.515 # NA = 1.44
f_0 = 10000         # overfilling
maxiter = 1000    # Maximum number of SA iterations
sa='no'          # Performs simulated annealing. If not, reads data from disk



#%% 

"""
Step 2: simmulated Annealing
"""

print("Step 2: simmulated annealing")

lw = [-1] * nterms # SA lower bounds
up = [1] * nterms  # SA upper bounds

r = np.linspace(1e-10, zonaVisPp2, NPP2) # radial variable @ focal plane


target = special.jv(1, n2 * coef * r)**2 #* np.exp(-r**2 / 2 / (zonaVisPp2 / 3)**2)
target = target / target.max()


if sa == 'yes':
    ret = differential_evolution(costFunc, bounds=list(zip(lw, up)), maxiter=maxiter)
    #ret = dual_annealing(costFunc, bounds=list(zip(lw, up)), maxiter=maxiter)
    
    coeficients = ret.x
    coeficients[0::2]=0
    print(coeficients)
    np.save('./coeficients_NA='+str(a)+'_coef='+str(coef)+'_NP='+str(NP)+'_nterms='+str(nterms)+'_zonaVisPp2='+str(zonaVisPp2)+'_limit='+str(up[0])+'_f0='+str(f_0)+'.npy', coeficients)


else:
    coeficients =\
        np.load('./coeficients_NA='+str(a)+'_coef='+str(coef)+'_NP='+str(NP)+'_nterms='+str(nterms)+'_zonaVisPp2='+str(zonaVisPp2)+'_limit='+str(up[0])+'_f0='+str(f_0)+'.npy')


perfilIntensitat = inten(coeficients) # Intensity as in Eq. (8), after SA


tic = time.time()
print('Time =', "{:.2f}".format((tic-toc)/60),"'")



#%%

"""
Step 3: profile distribution at the Entrance Pupil
"""

print("Step 3: profile at the EP")
rho = np.linspace(1e-10,1-1e-10, NP) # radial variable @ Gaussian sphere of reference

epsilon = np.zeros(NP)

for n in range(1, coeficients.size, 2):
     coefZR = coeffZernikeRadial_nm(n, 1)[::-1]    
     epsilon += coeficients[n] * np.polyval(coefZR, rho)
     
epsilon = - epsilon
rhoxg = epsilon * (rho <=a) * (1-rho**2)**0.25 


X, Y = np.meshgrid(np.linspace(-1, 1, NP), np.linspace(-1, 1, NP))
d = np.sqrt(X*X + Y*Y)


rhoxgPolar = np.tile(rhoxg, NP).reshape(NP,NP)

rhoxg2D = cv2.linearPolar(rhoxgPolar, (NPP2, NPP2), NPP2, cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS)
rhoxg2D = rhoxg2D * (d<1) 



#eff = (np.abs(rhoxg2D/np.abs(rhoxg2D).max())**2).sum()/(d<1).sum()
#print('Efficiency = ', eff )

presmall = np.copy(rhoxg2D)
small = rhoxg2D[::8, ::8] 
rhoxg2D = np.kron(small, np.ones((8, 8)))




plt.figure(figsize=(10,4.5))
plt.subplot(121)
plt.plot(rho, epsilon, label=r'$\epsilon(\rho)$')
plt.plot(rho[:], rhoxg[:], label=r'$\rho g(\rho$)')

plt.plot([a, a], [-1, 1], 'k-.')
plt.text(0.7, 0.5, str('a = '+str(a)), fontsize=13)
plt.legend()
#plt.grid()
plt.xlabel(r'$\rho$ (normalized coordinates)')
plt.ylim(-1,1)

plt.subplot(122)
plt.imshow(presmall, vmin=-1, vmax=1, extent=(-a, a, -a, a), cmap='seismic')
plt.xlabel(r'$\rho_x$ (normalized coordinates)')
plt.ylabel(r'$\rho_y$ (normalized coordinates)')
plt.colorbar()
plt.tight_layout()
plt.show()

valhol = np.zeros([NP, 3])
valhol[:, 0] = rho
valhol[:, 1] = epsilon
valhol[:, 2] = rhoxg

np.save('FigureE.npy', valhol)
plt.savefig('Hologram_profile____NA='+str(a)+'_coef='+str(coef)+'_NP='+str(NP)+'_nterms='+str(nterms)+'_zonaVisPp2='+str(zonaVisPp2)+'_limit='+str(up[0])+'_f0='+str(f_0)+'.png')
plt.savefig('FigureE.pdf')




#%%

"""
Step 4: integration using Eq. (6)
"""

print("Step 4: integration")
integral_r = np.zeros(NPP2)
integral_a = np.zeros(NPP2)

    
for r0, ii in zip(np.linspace(0, zonaVisPp2, NPP2), range(NPP2)):
    integral_r[ii] = np.abs(integrate.quad(cin, 0, 1, args=(r0))[0])**2
    integral_a[ii] = np.abs(integrate.quad(gaz, 0, .999, args=(r0))[0])**2 #the integral might diverge at \rho=1


integral_r = integral_r / integral_r.max()
integral_a = integral_a / integral_a.max()

#%%
plt.figure(figsize=(9,7.5))

plt.subplot(211)
plt.plot(r[:NPP2], integral_a[:NPP2], 'r-', label='Azimuthally polarized beam')
plt.plot(r[:NPP2], target[:NPP2], 'b-', label=r'Target function: $|\mathrm{J}_1(snr)|^2$') 
plt.plot(r[:NPP2], integral_r[:NPP2], 'g-', label='Designed beam')
plt.xlim(0,.4)
plt.xlabel(r'r in $\lambda$ units')
plt.ylabel(r'I (a.u.)')
max_r = round(r[int(np.argmax(integral_a))], 3)
max_t = round(r[int(np.argmax(target))], 3)
plt.plot([max_r, max_r], [0, 1], 'k-.')
plt.plot([max_t, max_t], [0, 1], 'k-.')
plt.text(0.17, 0.2, str(max_t), fontsize=13)
plt.text(0.24, 0.2, str(max_r), fontsize=13)
#plt.grid()
plt.legend()

plt.subplot(223)
imdes = np.tile(integral_r, NPP2).reshape(NPP2,NPP2)
imdes2D = cv2.linearPolar(imdes, (NPP4, NPP4), NPP4, cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS)
plt.imshow(imdes2D, cmap='hot', extent=(-zonaVisPp2, zonaVisPp2, -zonaVisPp2, zonaVisPp2))
plt.title('Designed beam')

plt.subplot(224)
imaz = np.tile(integral_a, NPP2).reshape(NPP2,NPP2)
imaz2D = cv2.linearPolar(imaz, (NPP4, NPP4), NPP4, cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS)
plt.imshow(imaz2D, cmap='hot', extent=(-zonaVisPp2, zonaVisPp2, -zonaVisPp2, zonaVisPp2))
plt.title('Azimuthally polarized beam')
plt.tight_layout()



valors = np.zeros([NPP2, 4])
valors[:, 0] = r
valors[:,1] = integral_a
valors[:,2] = target
valors[:,3] = integral_r

np.save('FigureA.npy', valors)
plt.savefig('Transverse_irradian_NA='+str(a)+'_coef='+str(coef)+'_NP='+str(NP)+'_nterms='+str(nterms)+'_zonaVisPp2='+str(zonaVisPp2)+'_limit='+str(up[0])+'_f0='+str(f_0)+'.svg')
#%%

plt.figure(figsize=(9,7.5))

plt.subplot(211)
plt.plot(r[:NPP2], integral_a[:NPP2], 'r-', label='Azimuthally polarized beam')
plt.plot(r[:NPP2], target[:NPP2], 'b-', label=r'Target function: $|\mathrm{J}_1(snr)|^2$') 
plt.plot(r[:NPP2], integral_r[:NPP2], 'g-', label='Designed beam')
plt.xlim(0,.4)
plt.xlabel(r'r in $\lambda$ units')
plt.ylabel(r'I (a.u.)')
max_r = round(r[int(np.argmax(integral_a))], 3)
max_t = round(r[int(np.argmax(target))], 3)
plt.plot([max_r, max_r], [0, 1], 'k-.')
plt.plot([max_t, max_t], [0, 1], 'k-.')
plt.text(0.17, 0.2, str(max_t), fontsize=13)
plt.text(0.24, 0.2, str(max_r), fontsize=13)
#plt.grid()
plt.legend()

plt.subplot(223)
imdes = np.tile(integral_r, NPP2).reshape(NPP2,NPP2)
imdes2D = cv2.linearPolar(imdes, (NPP4, NPP4), NPP4, cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS)
plt.imshow(imdes2D[NPP8:NPP8+NPP4, NPP8:NPP8+NPP4], cmap='hot', extent=(-zonaVisPp2//2, zonaVisPp2//2, -zonaVisPp2//2, zonaVisPp2//2))
plt.title('Designed beam')

plt.subplot(224)
imaz = np.tile(integral_a, NPP2).reshape(NPP2,NPP2)
imaz2D = cv2.linearPolar(imaz, (NPP4, NPP4), NPP4, cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS)
plt.imshow(imaz2D[NPP8:NPP8+NPP4, NPP8:NPP8+NPP4], cmap='hot', extent=(-zonaVisPp2//2, zonaVisPp2//2, -zonaVisPp2//2, zonaVisPp2//2))
plt.title('Azimuthally polarized beam')
plt.tight_layout()


plt.savefig('FigureA.pdf')

sys.exit()

#%%

"""
Step 5: propagation: calculation along the z-axis
"""

print("Step 5: propagation")

NPah = NP #// 8
NPahPp2 = NPah //2
NPZ = 31         # total steps for propagation
#NPzPp2 = NPZ // 2 +1
zlim = 20       # calculation of propagation [-zlim, zlim]
zonaVisPp2P = 8

integral_rz_R = np.zeros([NPahPp2, NPZ])
integral_rz_I = np.zeros([NPahPp2, NPZ])
integral_az_R = np.zeros([NPahPp2, NPZ])
integral_az_I = np.zeros([NPahPp2, NPZ])
#integral_rz_R = np.zeros([NPahPp2, NPzPp2])
#integral_rz_I = np.zeros([NPahPp2, NPzPp2])
#integral_az_R = np.zeros([NPahPp2, NPzPp2])
#integral_az_I = np.zeros([NPahPp2, NPzPp2])

# for z, jj in tqdm(zip(np.linspace(-zlim, 0, NPzPp2), range(NPzPp2)), total=NPzPp2):  
#     #print (jj, z)
#     for r0, ii in zip(np.linspace(0, zonaVisPp2P, NPahPp2), range(NPahPp2)):
#         integral_rz_R[ii, jj] = np.abs(integrate.quad(cinzr, 0, 1, args=(r0, z))[0])**2
#         integral_rz_I[ii, jj] = np.abs(integrate.quad(cinzi, 0, 1, args=(r0, z))[0])**2
        
#         #integral_az_R[ii, jj] = np.abs(integrate.quad(gazzr, 0, a, args=(r0, z))[0])**2
#         #integral_az_I[ii, jj] = np.abs(integrate.quad(gazzr, 0, a, args=(r0, z))[0])**2 

for z, jj in tqdm(zip(np.linspace(-zlim, zlim, NPZ), range(NPZ)), total=NPZ):  
    #print (jj, z)
    for r0, ii in zip(np.linspace(0, zonaVisPp2P, NPahPp2), range(NPahPp2)):
        integral_rz_R[ii, jj] = np.abs(integrate.quad(cinzr, 0, 1, args=(r0, z))[0])**2
        integral_rz_I[ii, jj] = np.abs(integrate.quad(cinzi, 0, 1, args=(r0, z))[0])**2
        
        integral_az_R[ii, jj] = np.abs(integrate.quad(gazzr, 0, a, args=(r0, z))[0])**2
        integral_az_I[ii, jj] = np.abs(integrate.quad(gazzr, 0, a, args=(r0, z))[0])**2 
        
integral_rz = np.sqrt(integral_rz_R**2 + integral_rz_I**2) #/ integral_r.max()
integral_az = np.sqrt(integral_az_R**2 + integral_az_I**2) #/ integral_a.max()


irz2 = np.concatenate((integral_rz[::-1,:],integral_rz[1::,:]), axis=0)
iaz2 = np.concatenate((integral_az[::-1,:], integral_az[1::,:]), axis=0)

#temp = irz2[:,::-1]
#irz2 = np.concatenate((irz2, temp[:,1:]), axis=1)
#temp = iaz2[:,::-1]
#iaz2 = np.concatenate((iaz2, temp[:,1:]), axis=1)


tic = time.time()
print('Time =', "{:.2f}".format((tic-toc)/60),"'")

#%%

"""
Step 6: results display in z
"""

print("Step 6: results")

plt.figure(figsize=(10,5))

plt.subplot(241)
plt.plot(r[:NPP2], integral_a[:NPP2], 'r-', label='GS')
plt.plot(r[:NPP2], target[:NPP2], 'b-', label='Target')
plt.plot(r[:NPP2], perfilIntensitat[:NPP2], 'g-', label='Calculated')
plt.xlim(0,1)
plt.xlabel(r'r ($\lambda$)')
plt.ylabel(r'I (a.u.)')
plt.grid()
plt.legend()

plt.subplot(242)
plt.plot(rho, epsilon, label=r'$\epsilon(\rho)$')
plt.plot(rho, rhoxg, label=r'$\rho g(\rho$)')
plt.legend()
plt.grid()
plt.xlabel(r'$\rho$ (normalized coordinates)')
plt.ylim(-2,2)

plt.subplot(243)
plt.imshow(rhoxg2D, vmin=-2, vmax=2, extent=(-a, a, -a, a), cmap='seismic')
plt.xlabel(r'$\rho_x$ (normalized coordinates)')
plt.ylabel(r'$\rho_y$ (normalized coordinates)')
plt.colorbar(shrink=0.8, location='bottom')

plt.subplot(244)
plt.plot(r[:NPP2], integral_a[:NPP2], 'r-', label='GS')
plt.xlim(0,1)
plt.plot(r[:NPP2], target, 'b-', label='Target')
plt.plot(r[:NPP2], integral_r[:NPP2], 'g-.', label='Calculated')
plt.xlabel(r'r ($\lambda$)')
plt.ylabel(r'I (a.u.)')
plt.grid()
plt.legend()

#%%
plt.figure(figsize=(14,10))
plt.subplot(211)
plt.imshow(irz2, cmap='hot', aspect=0.75, extent=(-zlim, zlim, -zonaVisPp2P, zonaVisPp2P))
plt.xlabel(r'z ($\lambda$)')
plt.ylabel(r'r ($\lambda$)')
plt.title("Designed beam")


#iaz2 = np.load('azimutal.npy')
plt.subplot(212)
plt.imshow(iaz2, cmap='hot', aspect=0.75, extent=(-zlim, zlim, -zonaVisPp2P, zonaVisPp2P))
plt.xlabel(r'z ($\lambda$)')
plt.ylabel(r'r ($\lambda$)')
plt.title("Azimuthally polarized beam")

plt.tight_layout()


plt.savefig('Evolution_two_beams_NA='+str(a)+'_coef='+str(coef)+'_NP='+str(NP)+'_nterms='+str(nterms)+'_zonaVisPp2='+str(zonaVisPp2)+'_limit='+str(up[0])+'_f0='+str(f_0)+'.png')
np.save('Evolution_designed__NA='+str(a)+'_coef='+str(coef)+'_NP='+str(NP)+'_nterms='+str(nterms)+'_zonaVisPp2='+str(zonaVisPp2)+'_limit='+str(up[0])+'_f0='+str(f_0)+'.npy', irz2)
np.save('Evolution_azimuthal_NA='+str(a)+'_coef='+str(coef)+'_NP='+str(NP)+'_nterms='+str(nterms)+'_zonaVisPp2='+str(zonaVisPp2)+'_limit='+str(up[0])+'_f0='+str(f_0)+'.npy', iaz2)

#%%




"""
izr2_0 = np.tile(irz2[:, 16], NP).reshape(NP,NP-1)

izr2D = cv2.linearPolar(izr2_0, (NPP2, NPP2), NPP2, cv2.WARP_INVERSE_MAP+cv2.WARP_FILL_OUTLIERS)

plt.figure()
plt.subplot(121)
plt.imshow(izr2D, cmap="hot", extent=(-zonaVisPp2P, zonaVisPp2P,-zonaVisPp2P, zonaVisPp2P))
plt.subplot(122)
plt.plot(np.linspace(-zonaVisPp2P, zonaVisPp2P, NP-1), irz2[:, 16] / irz2[:,16].max())
plt.grid()
plt.tight_layout()
"""