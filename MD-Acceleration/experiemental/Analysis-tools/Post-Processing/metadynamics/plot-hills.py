'''Plot the Gaussian height as a function of time for a well-tempered Metadynamics simulation'''
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple
from scipy.interpolate import griddata
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import os
import sys
import MDAnalysis as mda
import math
from numpy.linalg import norm
print("Required packages are imported")

#Color dictionaries: col1-Wes Anderson, col2-colorbrewer colorblind-safe
col1 = {'red': (0.58, 0.29, 0.31) , 'yellow': (0.76, 0.56, 0.00), 'blue': (0.38, 0.54, 0.60), 'green': (0.43, 0.60, 0.48)}
col2 = {'red': '#e66101' , 'orange': '#fdb863', 'lavender': '#b2abd2', 'purple': '#5e3c99'}
print("Colour dictionaries are defined")

large_font=14
small_font=12

Time=np.loadtxt(f"HILLS")[:, 0]/1000
GH=np.loadtxt(f"HILLS")[:, 5]

xmin=float("{0:.1f}".format(np.amin(Time)))              
xmax=float("{0:.1f}".format(np.amax(Time)))
ymin=float("{0:.1f}".format(np.amin(GH)))
ymax=float("{0:.1f}".format(np.amax(GH)))            

print(f"x-axis minimum and maximum is {xmin} and {xmax}")
print(f"y-axis minimum and maximum is {ymin} and {ymax}")
              
              
fig, (ax1) = plt.subplots(1,figsize=(8,4))

ax1.plot(Time[::20], GH[::20], color=col1['blue'], linewidth=1.0)
ax1.set_xlim([xmin, xmax])
ax1.set_ylim([ymin-0.1, ymax+0.1])
ax1.xaxis.set_ticks(np.arange(int(xmin), int(xmax), 50))
ax1.set_yticks(np.arange(ymin, ymax + 0.1, 0.1))
ax1.tick_params(axis='both', bottom=True, top=True, right=True, which='major', labelbottom=True, labeltop=False, direction='in', labelsize=small_font)
ax1.set_xlabel('Time [ps]', fontsize=large_font)
ax1.set_ylabel('Gaussian Height [kJ/mol]', fontsize=large_font)
# ax1.legend(leg, loc=1,  fancybox=True, shadow=True)

fig.tight_layout()

#plt.savefig(f"gaussian-height.png", bbox_inches='tight', dpi=600)
plt.show()
#plt.close()
