from __future__ import print_function
import sys
import numpy as np
cimport numpy as cnp
from libc.stdlib cimport strtol, malloc, free
from libc.stdint cimport int16_t, INT16_MAX
from libc.stdio cimport fopen, fseek, ftell, fclose, feof, getline, SEEK_SET, FILE


DTYPE = np.int16
ctypedef int16_t DTYPE_t
cdef size_t DTYPE_MAX = INT16_MAX


cdef size_t read_first_line(FILE *file, char **line_ptr, size_t *n_ptr,
                            cnp.ndarray[DTYPE_t, ndim=1, mode='c'] image):
    cdef char *current
    cdef char *end
    cdef size_t index
    cdef long int value
    assert image.size > 0
    if getline(line_ptr, n_ptr, file) <= 0:
        print('EOF')
        return 0
    current = line_ptr[0]
    value = strtol(current, &end, 10)
    if current == end:
        return 0
    assert value == 1
    index = 0
    while current != end:
        current = end
        value = strtol(current, &end, 10)
        if current == end:
            break
        image[index] = value
        index += 1
        if index >= image.size:
            image.resize((image.size + 1,), refcheck=False)
    return index


cdef read_other_line(FILE *file, size_t num_cols, size_t row_number,
                     char **line_ptr, size_t *n_ptr,
                     cnp.ndarray[DTYPE_t, ndim=1, mode='c'] image):
    cdef char *current
    cdef char *end
    cdef size_t index
    cdef size_t offset
    cdef long int value
    getline(line_ptr, n_ptr, file)
    current = line_ptr[0]
    value = strtol(current, &end, 10)
    assert value == <long int>(row_number), 'expecting first value %d, got %d' % (row_number, value)
    offset = (row_number - 1) * num_cols
    for index in range(num_cols):
        current = end
        value = strtol(current, &end, 10)
        if current == end:
            print('WARNING: too short line at row:', row_number)
            break
        image[offset + index] = value


cdef class cAscFile:
    cdef FILE *_file
    cdef size_t _num_rows
    cdef size_t _image_index
    cdef cnp.ndarray _image
    cdef object _image_start
    cdef char *_line_buffer
    cdef size_t _line_buffer_length
    def __cinit__(self, path, int num_rows=2048, dtype=np.int16):
        self._file = fopen(path.encode(sys.getdefaultencoding()), 'rb')
        if self._file == NULL:
            raise RuntimeError('could not open: %s' % path)
        self._num_rows = num_rows
        self._image_index = 0
        self._image = np.zeros((num_rows * num_rows,), dtype=dtype)
        self._image_start = []
        self._line_buffer_length = 8 * 1024
        self._line_buffer = <char *>malloc(self._line_buffer_length)

    def get_start_indices(self):
        return self._image_start

    def __iter__(self):
        self._image_index = 0
        self._image_start = []
        fseek(self._file, 0, SEEK_SET)
        return self

    def __next__(self):
        cdef long initial_position
        cdef size_t num_cols
        cdef size_t row_number
        if feof(self._file):
            raise StopIteration
        initial_position = ftell(self._file)
        num_cols = read_first_line(self._file, &self._line_buffer, &self._line_buffer_length, self._image)
        if num_cols == 0:
            raise StopIteration
        self._image_start.append(initial_position)
        self._image_index += 1
        if self._num_rows * num_cols != self._image.size:
            self._image.resize((self._num_rows * num_cols,), refcheck=False)
        for row_number in range(2, self._num_rows + 1):
            read_other_line(self._file, num_cols, row_number, &self._line_buffer, &self._line_buffer_length, self._image)
        return np.reshape(self._image, (self._num_rows, num_cols))

    def __dealloc__(self):
        #print('image start:', self._image_start)
        free(self._line_buffer)
        fclose(self._file)