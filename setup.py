import os
from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


extensions = [
    Extension('scs_xes.csm.cluster',
        ['scs_xes/csm/cluster.pyx']),
]

cython_kwargs = dict()
if int(os.getenv('profile', '0')):
    print('profile build')
    cython_kwargs['profile'] = True
    cython_kwargs['linetrace'] = True
if int(os.getenv('unsafe', '0')):
    print('aggressive optimizations')
    cython_kwargs['wraparound'] = False
    cython_kwargs['boundscheck'] = False

setup(
    name='scs_xes',
    description='X-ray emission spectroscopy software for FS-SCS',
    author='Mirko Scholz',
    version='0.2-dev',
    ext_modules=cythonize(extensions, compiler_directives=cython_kwargs),
)

