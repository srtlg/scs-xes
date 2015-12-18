from distutils.core import setup
from Cython.Build import cythonize

module_csm = cythonize('cluster.pyx', gdb_debug=True)

setup(
    name='X-ray emission spectroscopy for FS-SCS',
    ext_modules=module_csm,
)

