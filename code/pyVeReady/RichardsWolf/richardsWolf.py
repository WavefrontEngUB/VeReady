#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  9 12:32:42 2023

@author: artur
"""
import numpy as np
#import matplotlib.pyplot as plt
#import numpy.fft as fft
#import scipy.special as special
#import cv2
import tabulate as t
import fbeams
import sys
from scipy import signal

import matplotlib
matplotlib.use('TkAgg')
"""
runfile('C:/Users/Artur/OneDrive - Universitat de Barcelona/WorkInProgress/BeamCalc/0905_bw.py', args='0 A', wdir='C:/Users/Artur/OneDrive - Universitat de Barcelona/WorkInProgress/BeamCalc', current_namespace=True)
['C:\\Users\\Artur\\OneDrive - Universitat de Barcelona\\WorkInProgress\\BeamCalc\\0905_bw.py', '0', 'A']
"""


print(sys.argv[:])


fileinput='doubleramp.npy'
fileinput='zero-pily.npy'
fileinput='0-pi_pi-2pi.npy'
fileinput='invbw.npy'




"""
**************
INICIALIZATION
**************
"""  

casecalc = 2
m = 1
NP = 200
zonaVisPp2 = 1.5 #(en lambdes)


NPP2 = NP // 2
NA = 1.333
n1 = 1
n2 = 1.34
theta_max = np.arcsin(NA / n2)
f_0 = 1
factor =  int(NP * n2 / (4 * NA * zonaVisPp2))

# Z5 = 0 # Z5-astX, Z8-comaX, Z9-trefoil, Z12-esf
# Z8 = 0
# Z9 = 0
# Z12 = 0

if casecalc==1:
    polaritzacio='_pl'
if casecalc==2:
    polaritzacio='_pc'
if casecalc==3:
    polaritzacio='_rd'
if casecalc==4:
    polaritzacio='_az'    
    

filenameoutput='convencional_'+str(NP)+'_'+str(zonaVisPp2)+polaritzacio+'_m='+str(m)+'.npy'



#Egs = np.load('convencional_'+str(NP)+'_'+str(zonaVisPp2)+'_'+polaritzacio+'.npy')
#Etarget = np.load('Erodademoli_'+str(NP)+'_'+str(zonaVisPp2)+'_'+polaritzacio+'.npy')
#IRM = np.load('Irodademoli_'+str(NP)+'_'+str(zonaVisPp2)+'.npy')

comment = 'rho x Gaussia azimutal m=0, ast=0.25, NA=1.332\n'

kernelList = ['File',                    #0
              'Gaussian linear',         #1
              'Gaussian circular',       #2
              'Gaussian radial',         #3
              'Gaussian azimuthal',      #4
              # 'Laguerre 01 linear',      #5
              # 'Laguerre 01 circular',    #6
              # 'Laguerre 01 radial',      #7
              # 'Laguerre 01 azimuthal',   #8
              # 'Bessel J1 linear',        #9
              # 'Bessel J1 circular',      #10  
              # 'Bessel J1 radial',        #11
              # 'Bessel J1 azimuthal',     #12
              # 'Needle azimuthal',        #13
              ]                   
             

kernelIt = kernelList[casecalc]


dades = [['NA = ', NA], ['Angle max = ', '{0:.2f}'.format(180*theta_max / np.pi)], ['n1 = ', n1], ['n2 = ', n2], 
         ['f_0 = ', f_0], ['',], ['NP = ', NP], ['FactAmp', factor], 
         ['Pixels efectius = ', NP*factor], ['Limit pp2 = ', zonaVisPp2],
         ['Kernel = ', kernelIt], ['Càrrega topològica, m = ', m]]#, ['Zernikes (Z5-astX, Z8-comaX, Z9-trefoil, Z12-esf) =', Z5, Z8, Z9, Z12]]

print(t.tabulate(dades))



"""
************
MAIN PROGRAM
************
"""

phi, theta = fbeams.definicioAngles(NP)

e1 = [-np.sin(phi), np.cos(phi), np.zeros([NP, NP])]
e2 = [np.cos(phi), np.sin(phi), np.zeros([NP, NP])]
e2p = [np.cos(theta)*np.cos(phi), np.cos(theta)*np.sin(phi), np.sin(theta)]
s = [np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), -np.cos(theta)]

Ei = fbeams.kernelDef(kernelIt, theta, phi, m, NP, f_0, theta_max, fileinput)#, Z5, Z8, Z9, Z12)
_, Efs = fbeams.RichardsWolf(Ei, e1, e2, e2p, n1, n2, theta, theta_max, factor, NP)

Efs = Efs * np.conj(fbeams.phaseShiftEstimation(NP, factor))

fh = 'n' #(y)es or (n)o
figs = fbeams.showFocField(Ei, Efs, NP, zonaVisPp2, fh)
#figs = fbeams.showFocField2(Ei, Efs, Etarget, Egs, NP, zonaVisPp2)
#figs = fbeams.showFocField2(Ei, Efs, IRM, Egs, NP, zonaVisPp2)

#fwhm, Ifprof, _ = fbeams.fwhm(Efs, NP, zonaVisPp2)


Eia = np.sqrt(np.abs(Ei[0,:,:])**2+np.abs(Ei[1,:,:])**2) 
Efa = np.abs(Efs[0,:,:])**2 + np.abs(Efs[1,:,:])**2 + np.abs(Efs[2,:,:])**2
trans = Eia.sum() / Eia.max() /  np.pi / NPP2**2


p2p = zonaVisPp2 * (signal.argrelmax(Efa[:, NPP2])[0][1] - signal.argrelmax(Efa[:, NPP2])[0][0] ) / NPP2

# pixels actius per NP = 128, 12644
print(r'Transmittance: ', 100*trans)
print(r'Peak-to-peak: ($\lambda$): ', p2p)
np.save(filenameoutput, Efs)
"""
if fh=='n':
    np.save(fileoutput, Efs)
    datalist = np.array([casecalc, NP, zonaVisPp2, NA, n1, n2, theta_max, f_0, factor, m])
     #fbeams.filehandling(datalist, Ifprof, figs, comment)
"""    







