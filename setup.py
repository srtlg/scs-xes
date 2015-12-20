from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension


extensions = [
    Extension('scs_xes.csm.cluster',
        ['scs_xes/csm/cluster.pyx']),
]

setup(
    name='scs_xes',
    description='X-ray emission spectroscopy for FS-SCS',
    author='Mirko Scholz',
    version='0.2-dev',
    ext_modules=cythonize(extensions),
)

