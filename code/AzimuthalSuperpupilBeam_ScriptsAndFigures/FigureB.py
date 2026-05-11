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


plt.style.use('default')
#plt.rcParams['font.size'] = 14
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
   
    return cin(x, r0) * np.cos(2 * np.pi * z * x) 

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
   
    return cin(x, r0) * np.sin(2 * np.pi * z * x) 


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
    az = x * np.exp(-x**2 / f_0**2 / a**2) * special.jv(1, 2*np.pi*n2*r0*x) * x / (1 - x**2)**0.25
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
        
    return gaz(x, r0) * np.cos(2 * np.pi * z * x) 
           

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
        
    return gaz(x, r0) * np.sin(2 * np.pi * z * x)


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

toc = time.time()

print("Step 1: parameter definition")
NP = 2048   # Number of samples 
NPP2 = NP // 2    # Half number of samples
zonaVisPp2 = 6 #8    # Radius of the Visualization (in \lambdas) 
nterms = 8  # number of Zernike polynomials terms used (degree; 0, 2, 4,...) 
coef = 6        # J1 function scale factor; coef=5: argmax equivalent to GS, coef=6: superesolution, coef=7, does not converge
a = 0.9373          # a = sin(theta_{max})^2 / n_2 (NA in air)
n2 = 1.515          
f_0 = 10000        # overfilling



lw = [-1] * nterms # SA lower bounds
up = [1] * nterms  # SA upper bounds

r = np.linspace(1e-10, zonaVisPp2, NPP2) # radial variable @ focal plane
rho = np.linspace(1e-10,1-1e-10, NP) # radial variable @ Gaussian sphere of reference

target = special.jv(1, n2*coef*r)**2
target = target / target.max()

plt.figure(figsize=(10,5))
#plt.subplot(122)

#plt.plot(r[:NPP2], target[:NPP2], label=r'$|\mathrm{J}_1(rs)|^2$')

for nterms in [18, 12, 8, 4]:
    coeficients =\
            np.load('coeficients_NA='+str(a)+'_coef='+str(coef)+'_NP='+str(NP)+'_nterms='+str(nterms)+'_zonaVisPp2='+str(zonaVisPp2)+'_limit='+str(up[0])+'_f0='+str(f_0)+'.npy')
    
    perfilIntensitat = inten(coeficients) # Intensity as in Eq. (8), after SA

    plt.plot(r[:NPP2], perfilIntensitat[:NPP2], label=r'$\mathrm{max}\{n\}=$'+str(nterms-1))


integral_a = np.zeros(NP)

rs = np.linspace(0, zonaVisPp2, NP)    
for r0, ii in zip(rs, range(NP)):
    integral_a[ii] = np.abs(integrate.quad(gaz, 0, a, args=(r0))[0])**2 #the integral might diverge at \rho=1

integral_a = integral_a / integral_a.max()

target = np.abs(special.jv(1,6*n2*rs))**2
target = target / target.max()

#plt.subplot(121)
plt.plot(rs, integral_a, 'k+-', label='Azimuthally Polarized')
plt.plot(rs, target, 'b.-', label='Target')
max_r = round(rs[int(np.argmax(integral_a))], 3)
max_t = round(rs[int(np.argmax(target))], 3)
plt.plot([max_r, max_r], [0, 1], 'k-.')
plt.plot([max_t, max_t], [0, 1], 'k-.')
plt.text(0.195, 1.005, '0.205', fontsize=13)
plt.text(0.245, 1.005, '0.235', fontsize=13)

plt.xlim(0.15,.325)
plt.ylim(0.9,1.02)
plt.xlabel(r'r (in $\lambda$ units)')
plt.ylabel(r'I (a.u.)')
#plt.grid()
plt.legend()

#plt.plot(rs, target, 'b', label=r'Target function $|\mathrm{J}_1(sr)|^2$')
#plt.plot([0.308,0.308], [0, 1], color='k', linestyle='-.')
#plt.plot([0.308,0.308], [0, 1], color='k', linestyle='-.')
#plt.plot([0.393,0.393], [0, 1], color='k', linestyle='-.')
#plt.text(0.21,0.2, '0.308', fontsize='large')
#plt.text(0.40,0.2,'0.393',  fontsize='large')
#plt.xlim(0,1)
#plt.ylim(0,1.02)
#plt.xlabel(r'r in $\lambda$ units')
#plt.ylabel(r'I (a.u.)')
#plt.grid()
plt.legend()

plt.tight_layout()

plt.savefig('FigureB.svg')


"""
rs[np.argmax(integral_a)]
Out[152]: 0.39276990718124083 
rs[np.argmax(target)]
Out[153]: 0.3077674645823156
"""