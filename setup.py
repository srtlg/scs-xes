import os
import numpy
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


def _force_rebuild(extensions):
    '''
    build --inplace --force does not re-cythonize the sources
    and w/o profiling enabled, during profile the cython-module
    may segfault
    '''
    for extension in extensions:
        assert isinstance(extension, Extension)
        for f in extension.sources:
            if f.endswith('.pyx'):
                cfile = f.replace('.pyx', '.c')
                if os.path.exists(cfile):
                    os.unlink(cfile)


extensions = [
    Extension('scs_xes.csm.cluster',
        ['scs_xes/csm/cluster.pyx']),
    Extension('scs_xes.c_asc_file',
        ['scs_xes/c_asc_file.pyx']),
]

cython_kwargs = dict()
if int(os.getenv('profile', '0')):
    print('profile build')
    cython_kwargs['profile'] = True
    cython_kwargs['linetrace'] = True
    _force_rebuild(extensions)
if int(os.getenv('unsafe', '0')):
    print('aggressive optimizations')
    cython_kwargs['wraparound'] = False
    cython_kwargs['boundscheck'] = False
    _force_rebuild(extensions)

setup(
    name='scs_xes',
    description='X-ray emission spectroscopy software for FS-SCS',
    author='Mirko Scholz',
    version='0.3-dev',
    ext_modules=cythonize(extensions, compiler_directives=cython_kwargs),
    include_dirs=[numpy.get_include()],
)

