from __future__ import print_function
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
from asc_file import AscFile
from csm.cluster import cluster_analysis


def _argparse():
    p = argparse.ArgumentParser()
    p.add_argument('infile')
    p.add_argument('-t', '--threshold', type=int, default=100)
    p.add_argument('-c', '--clustersize', type=int, default=6)
    p.add_argument('-I', '--interactive', action='store_true', default=False)
    return p.parse_args()


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
    for i in img:
        print('cluster analysis...')
        clusters = cluster_analysis(i, args.threshold)
        if args.interactive:
            plt.clf()
            plt.imshow(i, vmin=args.threshold, vmax=args.threshold+10, 
                cmap=cmap.gray_r, interpolation='nearest')
            plt.plot(clusters.xc, clusters.yc, 'ro')
            plt.draw()
            input('plot>>')
        print('reading...')


if __name__ == '__main__':
    main()
