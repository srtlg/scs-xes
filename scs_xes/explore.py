import argparse
import h5py
import matplotlib.pyplot as plt


class Cluster:
    def __init__(self, path):
        self.h5 = h5py.File(path, 'r')

    def __getitem__(self, item):
        return self.h5['/cluster'][item]


def cmd_hist_ec(args, cluster):
    plt.hist(cluster['ec'], bins=args.hist_nbin, range=(0, args.hist_range_max), log=1)
    plt.xlabel('accumulated charge (AU)')
    plt.xlim(0, args.hist_range_max)
    plt.show()


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('hist-ec',))
    p.add_argument('infile')
    g1 = p.add_argument_group('histogram')
    g1.add_argument('--hist-range-max', type=int, default=1024)
    g1.add_argument('--hist-nbin', type=int, default=256)
    return p.parse_args()


def main():
    args = _parse_args()
    cluster = Cluster(args.infile)
    globals()['cmd_%s' % args.command.replace('-', '_')](args, cluster)


if __name__ == '__main__':
    main()
