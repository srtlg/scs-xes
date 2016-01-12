"""
X-ray emission spectrum of water
--------------------------------

Extracted from Figure 4b of
`J. Phys. Chem. B 2014, 118, 9398−9403 <http://dx.doi.org/10.1021/jp504577a>`_

"""
import numpy as np

LN_2 = np.log(2.0)

def Gaussian(height, center, hwhm):
    return lambda x: height*np.exp(-LN_2*((x-center)/hwhm)**2)

def Constant(a):
    return lambda x: a

d_4 = 0.573290022139
d_5 = 526.426372602
d_6 = 0.438124894897
d_7 = 0.179940018584
d_8 = 520.40101906
d_9 = 1.51341603559
d_10 = 0.327932937285
d_11 = 524.822088959
d_12 = 2.0533479336
d_15 = 0.527544954325
d_16 = 525.501344969
d_17 = 0.454181072622
d_1 = 0.0188298840247

p_2 = Gaussian(d_4, d_5, d_6)
p_3 = Gaussian(d_7, d_8, d_9)
p_4 = Gaussian(d_10, d_11, d_12)
p_6 = Gaussian(d_15, d_16, d_17)
p_1 = Constant(d_1)

energy_range_eV = 515.0, 535.0

def spectrum(x):
    return sum((f(x) for f in (p_1, p_2, p_3, p_4, p_6)))