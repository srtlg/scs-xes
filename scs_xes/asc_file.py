import numpy as np
from os import SEEK_CUR, SEEK_SET


class AscFile(object):
    def __init__(self, file, num_rows=2048, dtype=np.int32):
        self._file = file
        self._num_rows = num_rows
        self._dtype = dtype

    def __iter__(self):
        if hasattr(self._file, 'seek'):
            self._file.seek(0, SEEK_SET)
        return self

    def __next__(self):
        if hasattr(self._file, 'seek'):
            if self._file.read(1) == '':
                raise StopIteration
            else:
                self._file.seek(-1, SEEK_CUR)
        arr = np.genfromtxt(self._file, self._dtype, max_rows=self._num_rows, delimiter='\t')
        if arr.size == 0:
            raise StopIteration
        else:
            assert (arr[:, 0] == np.arange(1, self._num_rows + 1)).all()
            return arr[:, 1:]