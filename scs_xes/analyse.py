from __future__ import print_function
import argparse
import logging
import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from scipy.optimize import curve_fit
from scs_xes.asc_file import AscFile
from scs_xes.csm.cluster import cluster_analysis, Cluster_dtype


_log = logging.getLogger('scs-xes.analyse')


class FitGaussian:
    def __init__(self, x, y):
        self.x = x.astype('f')
        self.y = y.astype('f')
        self.par_initial = self._estimate_parameters()
        self.par_opt = None
        self.pcov = None

    def _estimate_parameters(self):
        max_i = np.argmax(self.y)
        max_x = self.x[max_i]
        height = self.y[max_i]
        peak = np.nonzero(self.y >= height / 3)
        return height, max_x, np.sqrt(np.sum((self.x[peak] - max_x)**2)) / len(peak)

    def optimize(self):
        self.par_opt, self.pcov = curve_fit(self.model, self.x, self.y, p0=self.par_initial)

    def model(self, xdata, height, center, width):
        return height * np.exp(-0.5*((xdata - center)/width)**2)

    def get_model(self):
        if self.par_opt is None:
            return self.model(self.x, *self.par_initial)
        else:
            return self.model(self.x, *self.par_opt)

    @property
    def center(self):
        if self.par_opt is None:
            return self.par_initial[1]
        else:
            return self.par_opt[1]


def _argparse():
    p = argparse.ArgumentParser()
    p.add_argument('infile')
    p.add_argument('-t', '--threshold', type=int, default=100)
    p.add_argument('-N', '--restrict-number-images', type=int, default=None)
    p.add_argument('-I', '--interactive', action='store_true', default=False)
    p.add_argument('--clf', action='store_true', default=False)
    p.add_argument('--vmax-diff', type=int, default=10)
    p.add_argument('-C', '--get-curvature', action='store_true', default=False)
    p.add_argument('-c', '--curvature', default=None)
    p.add_argument('-O', '--pickle-clusters', default=None)
    g1 = p.add_argument_group('curvature')
    g1.add_argument('--accumulation-coordinate', choices=('fast', 'slow'), default='slow')
    g1.add_argument('--accumulation-start', type=int, default=None)
    g1.add_argument('--accumulation-stop', type=int, default=None)
    g1.add_argument('--accumulation-slice', type=int, default=16)
    g2 = p.add_argument_group('energy bins')
    g2.add_argument('--energy-start', type=int, default=None)
    g2.add_argument('--energy-stop', type=int, default=None)
    g2.add_argument('--energy-nbin', type=int, default=1024)
    return p.parse_args()


def _show_single_framge(args, clusters, image):
    if args.interactive:
        plt.clf()
        plt.imshow(image, vmin=args.threshold, vmax=args.threshold+args.vmax_diff,
            cmap=cmap.gray_r, interpolation='nearest')
        plt.plot(clusters.xc, clusters.yc, 'ro')
        plt.draw()
        input('plot>>')


def _accumulate_clusters(acc, incr):
    assert incr.dtype == Cluster_dtype
    if acc is None:
        return incr
    else:
        assert acc.dtype == incr.dtype
        old_size = acc.size
        ret = np.resize(acc, (old_size + incr.size,))
        ret[old_size:] = incr[:]
        return ret


def _determine_curvature_correction(args, clusters_acc):
    if args.accumulation_coordinate == 'slow':
        get_acc = lambda x: x['yc']
        get_energy = lambda x: x['xc']
    elif args.accumulation_coordinate == 'fast':
        get_acc = lambda x: x['xc']
        get_energy = lambda x: x['yc']
    else:
        raise AssertionError
    min_acc = args.accumulation_start if args.accumulation_start else np.min(get_acc(clusters_acc))
    min_acc = int(np.round(min_acc))
    max_acc = args.accumulation_stop if args.accumulation_stop else np.max(get_acc(clusters_acc))
    max_acc = int(np.round(max_acc))
    if args.interactive:
        if args.clf:
            plt.ion()
        plt.figure(1)
    bin_edges = None
    energy = None
    curvature_energy_max = []
    curvature_acc_coord = []
    for idx_acc in range(min_acc, max_acc, args.accumulation_slice):
        subset = np.nonzero(
            (get_acc(clusters_acc) >= idx_acc)
            & (get_acc(clusters_acc) < idx_acc + args.accumulation_slice))
        if bin_edges is None:
            kwargs = {'bins': args.energy_nbin}
            if args.energy_start and args.energy_stop:
                kwargs['range'] = args.energy_start, args.energy_stop
            hist, bin_edges = np.histogram(get_energy(clusters_acc)[subset], **kwargs)
            energy = (bin_edges[:-1] + bin_edges[1:]) / 2
        else:
            hist, _ = np.histogram(get_energy(clusters_acc)[subset], bins=bin_edges)
        f = FitGaussian(energy, hist)
        f.optimize()
        curvature_acc_coord.append(idx_acc + args.accumulation_slice / 2.0)
        curvature_energy_max.append(f.center)
        if args.interactive:
            if args.clf:
                plt.clf()
            plt.plot(energy, hist, 'b.-')
            plt.plot(energy, f.get_model(), 'r-', linewidth=3)
            plt.xlabel('energy (pixel)')
            plt.ylabel('intensity (counts)')
            plt.draw()
            if args.clf:
                input('slice[%d:%d]>' % (idx_acc, idx_acc + args.accumulation_slice))
    curvature_acc_coord = np.asarray(curvature_acc_coord)
    curvature_energy_max = np.asarray(curvature_energy_max)
    lin_par, lin_pcov = curve_fit(lambda x, a, b: a*x + b, curvature_acc_coord, curvature_energy_max,
        p0=(np.mean(np.diff(curvature_energy_max)), curvature_energy_max[0]))
    print('curvature:', lin_par)
    if args.interactive:
        plt.figure(2)
        plt.plot(curvature_acc_coord, curvature_energy_max, 'o')
        plt.plot(curvature_acc_coord, lin_par[0] * curvature_acc_coord + lin_par[1])
        plt.xlabel('accumulation coordinate (pixel)')
        plt.ylabel('max. energy coordinate (pixel)')
        if args.clf:
            input('...')
        else:
            plt.ioff()
            plt.show()


def _apply_curvature(args, clusters_acc, correction):
    if args.accumulation_coordinate == 'slow':
        get_acc = lambda x: x['yc']
        get_energy = lambda x: x['xc']
    elif args.accumulation_coordinate == 'fast':
        get_acc = lambda x: x['xc']
        get_energy = lambda x: x['yc']
    else:
        raise AssertionError
    assert len(correction) == 2
    energy = get_energy(clusters_acc) - (correction[0] * get_acc(clusters_acc) + correction[1])
    if args.interactive:
        plt.hist(energy, args.energy_nbin, histtype='step')
        plt.xlabel('corrected energy (pixel)')
        plt.ylabel('intensity')
        if args.energy_start:
            plt.xlim(xmin=args.energy_start)
        if args.energy_stop:
            plt.xlim(xmax=args.energy_stop)
        plt.ioff()
        plt.show()


def _save_clusters(args, path, cluster_acc):
    assert isinstance(cluster_acc, np.ndarray)
    assert cluster_acc.dtype == Cluster_dtype
    with h5py.File(path, 'w') as fout:
        fout['/threshold'] = args.threshold
        fout['/image-files'] = path.infile
        fout['/cluster'] = cluster_acc


def _process_clusters(args, clusters_acc):
    if args.get_curvature:
        _determine_curvature_correction(args, clusters_acc)
    elif args.curvature:
        correction = [float(i) for i in args.curvature.split(',')]
        _apply_curvature(args, clusters_acc, correction)
    elif args.pickle_clusters is not None:
        if args.pickle_clusters.endswith('.h5'):
            tail = ''
        else:
            tail = '.h5'
        _log.debug('saving clusters to %s', args.pickle_clusters + tail)
        _save_clusters(args, args.picle_clusters, clusters_acc)
    else:
        fields = sorted(Cluster_dtype.fields.items(), key=lambda x: x[1][1])
        for field in fields:
            print(field[0], ' ' if field is fields[-1] else '', end='')
        print('')
        for cluster in clusters_acc:
            print(*cluster)


def _load_images(args):
    if args.infile.endswith('asc'):
        return AscFile(args.infile, dtype=np.int16)
    elif args.infile.endswith('dat'):
        img = []
        for path in args.infile.split(','):
            img.append(np.loadtxt(path, dtype=np.int16))
        return img
    elif args.infile.endswith('npy'):
        return np.load(args.infile)
    else:
        raise NotImplementedError


def main():
    args = _argparse()
    img = _load_images(args)
    if args.interactive:
        plt.ion()
    if hasattr(img, 'dtype') and img.dtype == Cluster_dtype:
        _log.info('processing pickled clusters from %s...', args.infile)
        clusters_acc = img
    else:
        _log.info('reading [0]...')
        clusters_acc = None
        index = 0
        for i in img:
            _log.info('cluster analysis...')
            clusters = cluster_analysis(i, args.threshold)
            _show_single_framge(args, clusters, i)
            clusters_acc = _accumulate_clusters(clusters_acc, clusters)
            index += 1
            if args.restrict_number_images and index >= args.restrict_number_images:
                break
            _log.info('reading [%d]...', index)
    _process_clusters(args, clusters_acc)


if __name__ == '__main__':
    logging.basicConfig(level=0)
    main()
