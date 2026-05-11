
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 14:54:15 2025

@author: artur.carnicer
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
from scipy import integrate, special
from tqdm import tqdm



plt.style.use('default')
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 13

NP = 2048   # Number of samples 
NPP2 = NP // 2    # Half number of samples
NPP4 = NP // 4
zonaVisPp2 = 6 #8    # Radius of the Visualization (in \lambdas) 
nterms = 8  # number of Zernike polynomials terms used (degree; 0, 2, 4,...) 
coef = 6        # J1 function scale factor; coef=5: argmax equivalent to GS, coef=6: superesolution, coef=7, does not converge
a = 0.9373          # a = sin(theta_{max})^2 / n_2 (NA in air)
n2 = 1.515          
f_0 = 10000        # overfilling
f_0 = 10000   
lw = [-1] * nterms # SA lower bounds
up = [1] * nterms  # SA upper bounds


NPah = NP #// 8
NPahPp2 = NPah //2
NPZ = 31         # total steps for propagation
#NPzPp2 = NPZ // 2 +1
zlim = 20       # calculation of propagation [-zlim, zlim]

plt.close('all')
plt.interactive(True)
posicio = 1
plt.figure(figsize=(12,6))
plt.subplot(321)
resa = np.load('Evolution_azimuthal_NA=0.9373_coef=6_NP=2048_nterms=8_zonaVisPp2=6_limit=1_f0=10000.npy')

rows = resa.shape[0] 
cols = resa.shape[1]


plt.imshow(resa[rows//4:rows//4 + rows//2, cols//4 +1:cols//4 +1 + cols//2], cmap='hot', aspect=0.75, extent=(-zlim // 2, zlim // 2, -zonaVisPp2 // 2, zonaVisPp2 // 2))
plt.ylabel(r'r (in $\lambda$ units)')
plt.title('Azimuthally polarized beam')

for ordre in [2, 4, 8, 12, 18]:
    posicio+=1
    plt.subplot(3,2,posicio)
    plt.title(r'$\mathrm{max\left\{ n \right\}}$ = '+str(ordre -1))
    resi = np.load('Evolution_designed__NA='+str(a)+'_coef='+str(coef)+'_NP='+str(NP)+'_nterms='+str(ordre)+'_zonaVisPp2='+str(zonaVisPp2)+'_limit='+str(up[0])+'_f0='+str(f_0)+'.npy')
    plt.imshow(resi[rows//4:rows//4 + rows//2, cols//4+1:cols//4 +1+ cols//2], cmap='hot', aspect=0.75, extent=(-zlim // 2, zlim //2, -zonaVisPp2 // 2, zonaVisPp2 // 2))
    if posicio == 1 or posicio == 3 or posicio == 5 : 
        plt.ylabel(r'r (in $\lambda$ units)')
    if posicio == 6 or posicio == 5: 
        plt.xlabel(r'z (in $\lambda$ units)')
    #if posicio == 4: plt.xlabel(r'z (in $\lambda$ units)')    
    #plt.text(-18, 9, 'max{n} = '+str(ordre-1), fontsize=18, color='white')
     
plt.tight_layout()
plt.savefig('FigureD.pdf')