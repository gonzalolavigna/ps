#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 23:33:41 2019

@author: glavigna
"""

import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
from   scipy import signal


#Cierro todos los graficos por default.
plt.close('all')

#%%
def align_samples(input_vector,coeffs_len):
    columns = np.size(input_vector)-coeffs_len + 1
    rows    = coeffs_len
    A_H = np.zeros((rows,columns))
    #Populate matrix
    for j in np.arange(columns):
        for i in np.arange(rows):
            A_H[coeffs_len-1-i][j] = input_vector[i+j]
        #print("{}".format(A_H[:,j]))
    return A_H        
#%%   
coeffs_len = 100
vector_length = 3000
lags = np.linspace(coeffs_len-1,vector_length ,vector_length + 1 -coeffs_len ,endpoint=False)
n = np.linspace(0,5,vector_length ,endpoint=True)
#d_desired = np.sin(2*50*np.pi*n)  + np.sin(2*100*np.pi*n) + np.sin(2*150*np.pi*n) + np.sin(2*200*np.pi*n) + np.sin(2*250*np.pi*n) 
d_desired = np.sin(2*50*np.pi*n)
#vector_in = d  + np.random.normal(0,2,np.size(d))
vector_in = d_desired + 1*np.random.normal(0,1,np.size(n))


AH =align_samples(vector_in,coeffs_len);
A  = AH.transpose()
d_desired_slice = d_desired[coeffs_len-1::]

#print(AH)
#print(A)
AUTO_CORR = AH@A
CROSS_CORR = AH@d_desired_slice
W = inv(AUTO_CORR)@CROSS_CORR 

d_est = A@W

plt.plot(lags,d_est)
plt.plot(d_desired)
plt.plot(vector_in)


#Coeficiente A que para los FIR vale siempre 1 
a = np.array([1,0]);

#Obtenemos la respuestata en frecuencia
(w , h) = signal.freqz(W,a)
plt.figure(4)
plt.plot((w / (2 * np.pi)), 20 * np.log10(abs(h)), 'b')
plt.title('FILTRO A MAGNITUD')
plt.xlabel('Frecuencia [FS]')
plt.ylabel('Atenuacion [DB]')
plt.grid()
plt.show()

#%%
plt.close('all')
def align_samples_2(samples_vector,ncoeffs):
    #De tener 20 muestras y 4 Coeficientes del filtro, las columnas de AH son 17
    columns = np.size(samples_vector) - ncoeffs + 1
    rows    = ncoeffs
    AH     = np.zeros((rows,columns))
    for i in np.arange(columns):
        AH[:,i] =samples_vector[np.arange(ncoeffs+i-1,-1+i,-1)]
    return AH

def wiener_coeffs(AH,D):
    AUTO_CORR   = AH@AH.transpose()
    CROSS_CORR  = AH@D
    W           = inv(AUTO_CORR)@CROSS_CORR  
    return W

def minimum_mse(AH,D):
    DH = D.transpose()
    A =  AH.transpose()
    emin = DH@D - DH@A@inv(AH@A)@AH@D
    return emin
    
            
#Parametros de la señal
sigma   = 0.5
ncoeffs = 100
f0      = 50 
fs      = 1000
ts      = 1/fs
t_max   = 5
N  = t_max*fs

#
tt              = np.linspace(0,t_max,N,endpoint = False)
d_desired       = np.sin(2*f0*np.pi*tt + np.pi/2)
d_desired_slice = d_desired [ncoeffs-1:]
tt_slice        = tt[ncoeffs-1:]

input_vector = d_desired + np.random.normal(0,sigma,np.size(d_desired))
plt.figure()
plt.plot(tt,d_desired)

BH=align_samples_2(input_vector,ncoeffs)
W =wiener_coeffs(BH,d_desired_slice)
d_est = BH.transpose()@W
e_min = minimum_mse(BH,d_desired_slice)
print('E MINIMI:{}'.format(e_min))

plt.subplot(3,1,2)
plt.title('SEÑAL DESEADA + RUIDO')
plt.plot(tt[:300],input_vector[:300])
plt.subplot(3,1,1)
plt.title('SEÑAL DESEADA')
plt.plot(tt[:300],d_desired[:300])
plt.subplot(3,1,3)
plt.title('SEÑAL FILTRADA')
plt.plot(tt_slice[:300-ncoeffs],d_est[:300-ncoeffs])

e_min_ARRAY = []

#Calculemos el error MSE
for i in np.arange(4,1000):
    BH_BIS          = align_samples_2(input_vector,i)
    d_desired_slice = d_desired [i-1:]
    e_min_ARRAY.append(minimum_mse(BH_BIS,d_desired_slice))
    
    
a = [1]
(w , h) = signal.freqz(W,a)
plt.figure()
plt.plot((w / (2 * np.pi)), 20 * np.log10(abs(h)), 'b')
plt.title('FILTRO A MAGNITUD')
plt.xlabel('Frecuencia [FS]')
plt.ylabel('Atenuacion [DB]')
plt.grid()
plt.show()

#AUTO_CORR  = BH@B
#CROSS_CORR = BH@d_desired_slice
#W = inv(AUTO_CORR)@CROSS_CORR 

        
    

