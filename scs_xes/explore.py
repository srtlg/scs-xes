import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scs_xes.analyse import load_images


class ImagePathResolver:
    def image_for_index(self, index):
        raise NotImplementedError


class ImagePathResolverHDF5(ImagePathResolver):
    def __init__(self, h5):
        self.number_rows = None
        self.paths = self._load_image_paths(h5)
        if len(self.paths) > 1:
            raise NotImplementedError

    @staticmethod
    def _load_image_paths(h5):
        paths = []
        idx = 0
        group = h5['/image-files']
        while True:
            h5_node = '%02d' % idx
            if h5_node in group:
                paths.append(group[h5_node].value)
            else:
                break
            idx += 1
        return paths

    def image_for_index(self, index):
        loader = load_images(self.paths, asc_number_rows=self.number_rows)
        it = iter(loader)
        for i in range(index + 1):
            img = next(it)
        return img


class ImagePathResolverDummy(ImagePathResolver):
    def __init__(self, path):
        self.path = path
        self.number_rows = None

    def image_for_index(self, index):
        loader = load_images(self, self.path)
        it = iter(loader)
        for i in range(index + 1):
            img = next(it)
        return img


class Cluster:
    def __init__(self, path):
        self.h5 = h5py.File(path, 'r')
        self.threshold = self.h5['/threshold'].value
        self.cluster = self.h5['/cluster'].value
        self.image_paths = ImagePathResolverHDF5(self.h5)

    def __getitem__(self, item):
        return self.cluster[item]

    def get_image(self, cluster_index):
        idx = self.cluster[cluster_index]['image']
        return self.image_paths.image_for_index(idx)

    def print_index(self, index):
        print(self.cluster[index])

    def plot_index(self, index, figure):
        assert isinstance(figure, Figure)
        image = self.get_image(index)
        cluster = self.cluster[index]
        si = int(np.ceil(np.sqrt(cluster['nc']) * 1.5))
        ix = int(np.round(cluster['xc']))
        iy = int(np.round(cluster['yc']))
        hx = max(0, ix - si)
        hy = max(0, iy - si)
        jx = min(image.shape[1], ix + si + 1)
        jy = min(image.shape[0], iy + si + 1)
        ax = plt.subplot(121)
        plt.imshow(image[hy:jy, hx:jx], vmin=0, vmax=np.max(image),
                   interpolation='nearest')
        plt.plot(cluster['yc'] - hy, cluster['xc'] - hx, 'wo')
        plt.subplot(122, sharex=ax, sharey=ax)
        plt.imshow(image[hy:jy, hx:jx], vmin=self.threshold, vmax=self.threshold + 10,
                   interpolation='nearest')
        plt.plot(cluster['yc'] - hy, cluster['xc'] - hx, 'wo')


def cmd_hist_ec(args, cluster):
    plt.hist(cluster['ec'], bins=args.hist_nbin, range=(0, args.hist_range_max), log=1)
    plt.xlabel('accumulated charge (ADU)')
    plt.xlim(0, args.hist_range_max)
    plt.show()


def cmd_hist_nc(args, cluster):
    if args.hist_nbin > args.hist_range_max:
        args.hist_nbin = args.hist_range_max
    plt.hist(cluster['nc'], bins=args.hist_nbin, range=(0, args.hist_range_max), log=1)
    plt.xlabel('event size (pixel)')
    plt.xlim(0, args.hist_range_max)
    plt.show()


def cmd_browse_events(args, cluster):
    assert args.event_subset is not None
    cluster.image_paths.number_rows = args.event_number_rows
    expr = args.event_subset.replace('ec', 'cluster["ec"]').replace('nc', 'cluster["nc"]')
    sel = np.nonzero(eval(expr, dict(cluster=cluster)))[0]
    print('expression \'%s\'' % expr, 'matches', len(sel), 'events')
    plt.ion()
    fig = plt.figure()
    for i in sel:
        fig.clf()
        cluster.print_index(i)
        cluster.plot_index(i, fig)
        plt.draw()
        ans = input('quit? ')
        if ans == 'q':
            plt.ioff()
            return


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=('hist-ec', 'hist-nc', 'browse-events'))
    p.add_argument('infile')
    g1 = p.add_argument_group('histogram')
    g1.add_argument('--hist-range-max', type=int, default=1024)
    g1.add_argument('--hist-nbin', type=int, default=256)
    g2 = p.add_argument_group('browse events')
    g2.add_argument('--event-subset')
    g2.add_argument('--event-number-rows', type=int, default=2048)
    return p.parse_args()


def main():
    args = _parse_args()
    cluster = Cluster(args.infile)
    globals()['cmd_%s' % args.command.replace('-', '_')](args, cluster)


if __name__ == '__main__':
    main()
