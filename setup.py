import os, sys, os.path
from distutils.core import setup
from setuptools import setup, find_packages

setup(
    name='pycoss',
    version='0.0.1',
    url='https://github.com/lukastk/pycoss',
    author='Lukas Kikuchi',
    author_email='ltk26@cam.ac.uk',
    license='MIT',
    description='Cosserat rod simulations.',
    platforms='Tested on Linux',
    libraries=[],
    #packages=['pycoss', 'pycoss.rod', 'pycoss.rod.integrators', 'pycoss.relativistic_rod', 'pycoss.rod_on_sphere', 'pycoss.surface'],
    packages=find_packages()
)