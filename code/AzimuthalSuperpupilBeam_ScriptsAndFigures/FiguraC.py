# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:39:13 2025

@author: artur.carnicer
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import special

plt.style.use('default')
plt.rcParams['xtick.labelsize']=12
plt.rcParams['ytick.labelsize']=12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 14

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
    summand = np.zeros(NP)
    
    for n in range(0, alphas.size):
        if n%2 == 0:
            alphas[n]=0
        else:
            summand += alphas[n] * a**2 * (-1)**((n-1)/2) * np.sqrt(1) *\
                     special.jv(n+1, 2*np.pi*r*a) / (2*np.pi*r*a) # np.sqrt(n+1)

    res = np.abs(summand)**2
    return res / res.max()

NP=2048
a = 0.9373          # a = sin(theta_{max})^2 / n_2 (NA in air)
n2 = 1.515   




c10_2=np.load('coeficients_NA=0.9373_coef=6_NP=2048_nterms=8_zonaVisPp2=2_limit=1_f0=10000.npy')
c10_6=np.load('coeficients_NA=0.9373_coef=6_NP=2048_nterms=8_zonaVisPp2=6_limit=1_f0=10000.npy')
c20_2=np.load('coeficients_NA=0.9373_coef=6_NP=2048_nterms=18_zonaVisPp2=2_limit=1_f0=10000.npy')
c20_6=np.load('coeficients_NA=0.9373_coef=6_NP=2048_nterms=18_zonaVisPp2=6_limit=1_f0=10000.npy')


r = np.linspace(-12, 12, NP)
# r2 = np.linspace(-2, 2, NP)
# r3 = np.linspace(-6, 6, NP)
# r2: 1707 - 3088
# r3: 1024 - 3072


j1 = np.abs(np.abs(special.jv(1, 6*r)) / np.abs(special.jv(1, 6*r)).max())**2
#%%

plt.figure(figsize=(12,12))
plt.subplot(211)
plt.plot(r, inten(c10_2), label=r'$\mathrm{max}\{n\}= 7$, L=4$\lambda$')
plt.plot(r, inten(c20_2), label=r'$\mathrm{max}\{n\}= 17$, L=4$\lambda$')
plt.plot(r, j1, label=r'$|\mathrm{J}_1(sr)|^2$')
#plt.plot([-2, -2], [0,1], 'r')
#plt.plot([2, 2], [0,1], 'r')
plt.plot([-2, 2], [0,0], 'r', linewidth=4)
plt.text(-12, 0.9, "(a)", fontsize=18)
plt.xlabel(r'$r$ in $\lambda$ units')
plt.ylabel(r'$I(r)$ (a.u.)')
plt.legend()
#plt.grid()

# =============================================================================
# plt.figure()
# #plt.subplot(211)
# plt.plot(r2, inten(c10_2), label=r'n_$\mathrm{max}$=9, window=4$\lambda$')
# plt.plot(r2, inten(c20_2), label=r'n_$\mathrm{max}$=19, window=4$\lambda$')
# plt.plot(r2, j1, label=r'$|\mathrm{J}_1(sr)|^2$')
# plt.xlabel(r'$r$ in $\lambda$ units')
# plt.ylabel(r'$I(r)$ (a.u.)')
# plt.legend()
# plt.grid()
# =============================================================================

plt.subplot(212)
plt.plot(r, inten(c10_6), label=r'$\mathrm{max}\{n\}= 7$, L=12$\lambda$')
plt.plot(r, inten(c20_6), label=r'$\mathrm{max}\{n\}= 17$, L=12$\lambda$')
plt.plot(r, j1, label=r'$|\mathrm{J}_1(sr)|^2$')
#plt.plot([-6, -6], [0,1], 'r')
#plt.plot([6, 6], [0,1], 'r')
plt.plot([-6, 6], [0,0], 'r', linewidth=4)
plt.text(-12, 0.9, "(b)", fontsize=18)
plt.xlabel(r'$r$ in $\lambda$ units')
plt.ylabel(r'$I(r)$ (a.u.)')
plt.legend(fontsize=18)
#plt.grid()
plt.legend()
plt.tight_layout()

plt.savefig('FigureC.svg')

# =============================================================================
# 
# plt.figure()
# #plt.subplot(212)
# plt.plot(r3, inten(c10_6), label=r'n_$\mathrm{max}$=9, window=12$\lambda$')
# plt.plot(r3, inten(c20_6), label=r'n_$\mathrm{max}$=19, window=12$\lambda$')
# plt.plot(r3, j1, label=r'$\mathrm{J}_1(sr)$')
# plt.xlabel(r'$r$ in $\lambda$ units')
# plt.ylabel(r'$I(r)$ (a.u.)')
# plt.legend()
# plt.grid()
# plt.legend()
# plt.tight_layout()
# =============================================================================
