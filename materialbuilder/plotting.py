import numpy as np
import time
import matplotlib.pyplot as plt

SMALL_SIZE = 18
MEDIUM_SIZE = SMALL_SIZE*1.25
BIGGER_SIZE = SMALL_SIZE*1.5
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['figure.dpi'] = 300

CPK_COLORS = {
    0:  (np.array([0, 0, 0])/255).tolist(),
    1:  (np.array([255,255,255])/255).tolist(),
    2:  (np.array([217,255,255])/255).tolist(),
    3:  (np.array([204,128,255])/255).tolist(),
    4:  (np.array([194,255,0])/255).tolist(),
    5:  (np.array([255,181,181])/255).tolist(),
    6:  (np.array([144,144,144])/255).tolist(),
    7:  (np.array([48,80,248])/255).tolist(),
    8:  (np.array([255,13,13])/255).tolist(),
    9:  (np.array([144,224,80])/255).tolist(),
    10: (np.array([179,227,245])/255).tolist(),
    11: (np.array([171,92,242])/255).tolist(),
    12: (np.array([138,255,0])/255).tolist(),
    13: (np.array([191,166,166])/255).tolist(),
    14: (np.array([240,200,160])/255).tolist(),
    15: (np.array([255,128,0])/255).tolist(),
    16: (np.array([255,255,48])/255).tolist(),
    17: (np.array([31,240,31])/255).tolist(),
    18: (np.array([128,209,227])/255).tolist(),
    19: (np.array([143,64,212])/255).tolist(),
    20: (np.array([61,255,0])/255).tolist(),
    25: (np.array([156,122,199])/255).tolist(),
    26: (np.array([224,102,51])/255).tolist(),
    27: (np.array([240,144,160])/255).tolist(),
    28: (np.array([80,208,80])/255).tolist(),
    29: (np.array([200,128,51])/255).tolist(),
    30: (np.array([125,128,176])/255).tolist(),
    34: (np.array([255,161,0])/255).tolist(),
    35: (np.array([166,41,41])/255).tolist(),
    47: (np.array([192,192,192])/255).tolist(),
    53: (np.array([148,0,148])/255).tolist(),
    55: (np.array([87,23,143])/255).tolist(),
    79: (np.array([255,209,35])/255).tolist()
}

SYBYL_COLORS = {
    0:  CPK_COLORS[0],    # '?',
    1:  CPK_COLORS[1],    # 'H',
    2:  CPK_COLORS[6],    # 'C.3',
    3:  CPK_COLORS[6],    # 'C.2',
    4:  CPK_COLORS[6],    # 'C.1',
    5:  CPK_COLORS[6],    # 'C.ar',
    6:  CPK_COLORS[6],    # 'C.cat',
    7:  CPK_COLORS[7],    # 'N.3',
    8:  CPK_COLORS[7],    # 'N.2',
    9:  CPK_COLORS[7],    # 'N.1',
    10: CPK_COLORS[7],    # 'N.ar',
    11: CPK_COLORS[7],    # 'N.am0',
    12: CPK_COLORS[7],    # 'N.am',
    13: CPK_COLORS[7],    # 'N.am2',
    14: CPK_COLORS[7],    # 'N.pl3',
    15: CPK_COLORS[7],    # 'N.4',
    16: CPK_COLORS[8],    # 'O.3',
    17: CPK_COLORS[8],    # 'O.2',
    18: CPK_COLORS[8],    # 'O.co2',
    19: CPK_COLORS[16],   # 'S.3',
    20: CPK_COLORS[16],   # 'S.2',
    21: CPK_COLORS[16],   # 'S.o',
    22: CPK_COLORS[16],   # 'S.o2',
    23: CPK_COLORS[15],   # 'P.3',
    24: CPK_COLORS[9],    # 'F',
    25: CPK_COLORS[17],   # 'Cl',
    26: CPK_COLORS[34],   # 'Se',
    27: CPK_COLORS[35],   # 'Br',
    28: CPK_COLORS[53]    # 'I'
}
