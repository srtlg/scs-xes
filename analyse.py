from __future__ import print_function
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from asc_file import AscFile
from csm.cluster import cluster_analysis, Cluster_dtype


_log = logging.getLogger('scs-xes.analyse')


def _argparse():
    p = argparse.ArgumentParser()
    p.add_argument('infile')
    p.add_argument('-t', '--threshold', type=int, default=100)
    p.add_argument('-I', '--interactive', action='store_true', default=False)
    p.add_argument('--vmax-diff', type=int, default=10)
    p.add_argument('--curvature', action='store_true', default=False)
    p.add_argument('-O', '--pickle-clusters', default=None)
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
        acc.resize((old_size + incr.size,))
        acc[old_size:] = incr[:]
        return acc


def _process_clusters(args, clusters_acc):
    if args.curvature:
        pass
    elif args.pickle_clusters is not None:
        if args.pickle_clusters.endswith('npy'):
            tail = ''
        else:
            tail = '.npy'
        _log.debug('saving clusters to %s', args.pickle_clusters + tail)
        np.save(args.pickle_clusters, clusters_acc)
    else:
        k = list(Cluster_dtype.fields.items())
        # FIXME: how to sort a list properly
        #k.sort(lambda x: x[1][1])
        #for field in k:
        #    print(field)
        for cluster in clusters_acc:
            print(*cluster)


def main():
    args = _argparse()
    if args.infile.endswith('asc'):
        img = AscFile(args.infile, dtype=np.int16)
    elif args.infile.endswith('dat'):
        img = [np.loadtxt(args.infile, dtype=np.int16)]
    else:
        raise NotImplementedError
    if args.interactive:
        plt.ion()
    _log.info('reading...')
    clusters_acc = None
    for i in img:
        _log.info('cluster analysis...')
        clusters = cluster_analysis(i, args.threshold)
        _show_single_framge(args, clusters, i)
        clusters_acc = _accumulate_clusters(clusters_acc, clusters)
        _log.info('reading...')
    _process_clusters(args, clusters_acc)


if __name__ == '__main__':
    logging.basicConfig(level=0)
    main()
