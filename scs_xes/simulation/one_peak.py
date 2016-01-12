import numpy as np

LN_2 = np.log(2.0)
energy_range_eV = 100.0, 200.0


def spectrum(x):
    return np.exp(-LN_2 * ((x - 150.0) / 20.0)**2) + 0.01