#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 21:13:16 2019

@author: glavigna
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.ticker


#Cierro todos los graficos por default.
plt.close('all')

wp = 0.2
ws = 0.25
gpass = 0.1
gstop = 60

b,a = signal.iirdesign(wp, ws, gpass, gstop,output = 'ba')
w, h = signal.freqz(b,a)
 
fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response output = ba')
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')
ax1.grid()
 


system = signal.iirdesign(wp, ws, gpass, gstop,output = 'sos')
w, h = signal.sosfreqz(system)
 
fig, ax1 = plt.subplots()
ax1.set_title('Digital filter frequency response output = sos')
ax1.plot(w, 20 * np.log10(abs(h)), 'b')
ax1.set_ylabel('Amplitude [dB]', color='b')
ax1.set_xlabel('Frequency [rad/sample]')
ax1.grid()