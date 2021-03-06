"""
Use Python's profiler to speed up the cluster analysis
"""
import argparse
import cProfile
import numpy as np
from scs_xes.csm.cluster import cluster_analysis
from scs_xes.asc_file import AscFile


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('infile')
    p.add_argument('-t', '--threshold', type=int, default=80)
    return p.parse_args()


def _load_image(args):
    print('loading image...')
    if args.infile.endswith('.asc'):
        image = AscFile(args.infile)
        it = iter(image)
        return next(image)
    elif args.infile.endswith('.dat'):
        return np.loadtxt(args.infile, dtype=np.int16)


def main():
    args = _parse_args()
    image = _load_image(args)
    print('run cluster analysis...')
    cProfile.runctx('cluster_analysis(image, args.threshold)',
                    globals(), locals(), 'test_profile.prof')
    print('done.')


if __name__ == '__main__':
    main()
