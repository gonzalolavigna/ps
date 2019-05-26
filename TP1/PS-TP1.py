# -*- coding: utf-8 -*-
"""
Created on Mon May 13 20:12:45 2019

@author: glavigna
"""

#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sc
from scipy import signal

#%%

def simple_fft(yy,fs,N):
    """
    brief:  Genera la DFT, pero utilizando la FFT como algoritmo
    Entradas
    yy: Señal de entrada a convertir en DFT
    ts: Tiempo de sampleo en segundos
    N:  Numero de muestras de la señal
    
    Salidas
    ff: Campo de las frecuencias para poder hacer un grafico
    XX: Espectro de la señal en valor absoluto y solo una mitad.
        Magnitud Normalizada.
    """
    
    delta_f = (fs/2)/(N//2) ;
    
    XX = (2/N)*np.abs(sc.fft(yy));
    XX = XX[0:N//2];
    ff = np.linspace(0,(fs/2)-delta_f,N//2);
     
    return ff,XX

#Lo que hacemos es ver en un offset si mayor o menor que cero un valor.
def threshold_level_decision(vector,offset):
    if(vector[offset] > 0):
        return 1
    else :
        return 0


#%%
#Cierro todos los graficos por default.
plt.close('all')

#Abrimos el archivo correspondiente al pulso
pulse = np.load('pulse.npy')
pulse_repeat = np.tile(pulse,1)
#Dibujamos la señal correspondeinte.
plt.figure(1)
plt.plot(pulse_repeat)

N = len(pulse_repeat)
#Dibujamos la FFT, adaptado de la funcion de PSF con frecuencia de muestreo
#igual a 1.
pulse_fft_db = 20*np.log10(np.abs(sc.fft(pulse_repeat )))
pulse_fft    = np.abs(sc.fft(pulse_repeat ))
half_fft     = pulse_fft[0:N//2]
plt.figure(2)
plt.stem(half_fft)

#Sacamos la energia hasta tener el m
L = 5
energia        = np.sum(half_fft)
energia_slice  = np.sum(half_fft[0:L])
print("Porcentaje Energia: {} hast a muestra".format(energia_slice/energia))

#Abrimops el archivo correspondiente a la señal a la cual realizarle 
#el filtro
#signal_noise = np.load('signalLowSNR.npy')
signal_noise = np.load('signal.npy')

#Load low pass FIR
files = np.load('low_pass_filter.npz')
coefficient=files['ba.npy']
b = coefficient[0].flatten()
a = coefficient[1].flatten()

signal_filtered_lp_1 = signal.lfilter(b,a,signal_noise)
signal_filtered_lp_2 = signal.filtfilt(b,a,signal_noise)
#Load high pass FIR
files = np.load('high_pass_filter.npz')
coefficient=files['ba.npy']
b = coefficient[0].flatten()
a = coefficient[1].flatten()

signal_filtered_hp = signal.filtfilt(b,a,signal_filtered_lp_2)

plt.figure(4)
low_slice  = 0
high_slice = 160

plt.plot(signal_filtered_lp_2[low_slice:high_slice])


slice_1 = signal_filtered_lp_2[low_slice:high_slice]  - np.mean(signal_filtered_lp_2[low_slice:high_slice])
print("Valor en la mitad:{}".format(slice_1[4]))

bit_sequence = np.array([1,0,1,0,1,1,0,1])



#%%

#Load low pass FIR este filtro fue generado por la herramienta pyfda
#Cargamos el filtro digital para ver su transferencia.
#Podemos sacarle mucho ruido de esta manera
files = np.load('low_pass_filter.npz')
coefficient=files['ba.npy']
b = coefficient[0].flatten()
a = coefficient[1].flatten()

w, h = signal.freqz(b,a)

fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response output = ba')
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')
ax1.grid()

#%%


#Generamos a modo de ejemplo una secuencia correspondiente al header de la señal
#%%%

pulse = np.load('pulse.npy')

bit_samples  = 20
bit_in_bytes = 8
bytes_in_header = 16

first_slice  = bit_samples*bit_in_bytes
second_slice = 2*bit_samples*bit_in_bytes

header_bit_stream = np.concatenate((pulse, -pulse, pulse,-pulse,pulse,pulse,-pulse,-pulse), axis=0)
header_bit_stream = np.tile(header_bit_stream,bytes_in_header)
plt.figure(4)
plt.plot(header_bit_stream[first_slice:second_slice])
plt.title("Patron 8'b10101100 Byte en el header")
plt.grid()

#Aplicamos la tecnica que genera dos veces el filtrado de esta manera eliminada el retardo de grupo.
header_bit_stream_lp = signal.filtfilt(b,a,header_bit_stream)
plt.figure(5)
plt.plot(header_bit_stream_lp[first_slice:second_slice])
plt.title("Patron 8'b10101100 Byte en el header luego del filtrado propuesto")
plt.grid()

#%%%

#%%

offset = 2
low_slice  = 0
high_slice = 0 

bit_array = np.array([])

for i in range((bit_in_bytes*bytes_in_header)):
    low_slice = i*20;
    high_slice = (i+1)*20
    vector_slice = header_bit_stream_lp[low_slice : high_slice]
    #bit_array = np.concatenate((bit_array,threshold_level_decision(vector_slice,offset)),axis=0)
    bit_array=np.append(bit_array,threshold_level_decision(vector_slice,offset))
    


#%%



