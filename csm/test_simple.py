import numpy as np
from io import StringIO
from cluster import cluster_analysis
from test_cluster import TEST_IMAGE_2


def main():
    image = np.loadtxt(StringIO(TEST_IMAGE_2.replace('.', '0')), dtype=np.int16)
    clusters = cluster_analysis(image)
    for i, c in enumerate(clusters):
        print (i, c)


if __name__ == '__main__':
    main()
