*****************************************************
SCS-XES: X-ray emission spectroscopy software for SCS
*****************************************************


============
Installation
============

.. code-block::

	python3 setup.py build_ext --inplace


=====
Usage
=====


Cluster analysis
----------------

Analyze the first 30 frames (``-N30``) with a threshold of 70 (``-t70``) and save
the result to ``476-s30.h5``.

.. code-block::

	python -mscs_xes.analyse ../H2O_540eV_476.asc -t70 -N30 -O 476-s30


Determine the curvature
-----------------------

.. code-block::

    python -mscs_xes.analyse 476-s30.h5 -C \
    --energy-nbin=512 --accumulation-stop=1500 --accumulation-slice=64


Generate spectrum
-----------------

.. code-block::

    python -mscs_xes.analyse 476-s30.h5  -c-2.12726373e-01,1.18439527e+03 -I \
    --energy-nbin=4096 --energy-start=-100 --energy-stop=150


=======
License
=======

Copyright © 2015 Mirko Scholz

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

..
  vim:set spell spl=en:
