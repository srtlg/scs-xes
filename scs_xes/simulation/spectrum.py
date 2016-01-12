"""
Simulate a spectrum based on a continuous function
"""
import argparse
import importlib
import numpy as np
import numpy.polynomial.chebyshev as np_cheb
import matplotlib.pyplot as plt


INTERACTIVE1 = False


class Spectrum:
    """
    Generate X-ray photons with an energy distribution as the given spectrum

    Uses the transformation method of uniformly distributed random numbers
    (see [NumRecipPas]_ §7.2).

    In order to invert the corresponding CDF we use Chebyshev polynomials
    (see somewhere at SE).
    """
    def __init__(self, spectrum):
        mod = importlib.import_module('scs_xes.simulation.%s' % spectrum)
        self.spectrum = getattr(mod, 'spectrum')
        self.energy_range_eV = getattr(mod, 'energy_range_eV')
        self.trafo_resolution = 1000
        self.cheb_degree = 24
        self.cheb_coef = None
        self.energy_to_pixel = None

    def _prepare_transformation_method(self):
        energy = np.linspace(min(self.energy_range_eV), max(self.energy_range_eV), self.trafo_resolution)
        pdf_normed = self.spectrum(energy)
        pdf_normed /= np.sum(pdf_normed)
        cdf_normed = np.cumsum(pdf_normed)
        cdf_cheb_domain = 2 * cdf_normed - 1.0
        cheb_coef = np_cheb.chebfit(cdf_cheb_domain, energy, self.cheb_degree)
        if INTERACTIVE1:
            plt.figure(1)
            ax1 = plt.subplot(211)
            plt.plot(energy, cdf_normed)
            plt.subplot(212, sharex=ax1)
            plt.plot(energy, pdf_normed)
            plt.xlim(self.energy_range_eV)
            plt.figure(2)
            ax2 = plt.subplot(211)
            cdf_inverse = np_cheb.chebval(cdf_cheb_domain, cheb_coef)
            plt.plot(cdf_cheb_domain, energy)
            plt.plot(cdf_cheb_domain, cdf_inverse)
            plt.ylim(*self.energy_range_eV)
            plt.subplot(212, sharex=ax2)
            plt.plot(cdf_cheb_domain, energy - cdf_inverse)
            plt.xlim(-1.0, 1.0)
        return cheb_coef

    def simulate(self, num_events):
        if self.cheb_coef is None:
            self.cheb_coef = self._prepare_transformation_method()
        return np_cheb.chebval(2 * np.random.random_sample((num_events,)) - 1.0, self.cheb_coef)


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('spectrum', help='the spectrum to simulate', nargs='?', default='water')
    p.add_argument('-n', '--number-of-events', type=int, default=10000)
    p.add_argument('-b', '--number-of-bins', type=int, default=30)
    p.add_argument('-I', '--interactive', action='store_true', default=False)
    p.add_argument('-D', '--do-not-seed', action='store_true', default=False)
    g1 = p.add_argument_group('energy to pixel conversion')
    g1.add_argument('-s', '--eV-to-px-slope', type=float, default=None)
    g1.add_argument('-i', '--eV-to-px-intercept', type=float, default=0.0)
    return p.parse_args()


def main():
    args = _parse_args()
    sp = Spectrum(args.spectrum)
    if args.eV_to_px_slope:
        sp.energy_to_pixel = args.eV_to_px_slope, args.eV_to_px_intercept
    energies = sp.simulate(args.number_of_events)
    if not args.do_not_seed:
        np.random.seed()
    if args.interactive:
        plt.figure()
        plt.hist(energies, args.number_of_bins, histtype='step')
        plt.xlim(*sp.energy_range_eV)
    if any((INTERACTIVE1, args.interactive)):
        plt.show()


if __name__ == '__main__':
    main()