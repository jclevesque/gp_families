#############################################################
# Copyright (C) 2015 Audrey Durand, Julien-Charles Levesque
#
# Distributed under the MIT License.
# (See accompanying file LICENSE or copy at
#  http://opensource.org/licenses/MIT)
##############################################################

'''
Setup for gp_families project
'''

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='gp_families',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.1',

    description='Gaussian process families for CASH problems',
    long_description=long_description,

    # The project's main homepage.
    url='https://github.com/jclevesque/gp_families',

    # Author details
    author='Audrey Durand',
    author_email='audrey.durand.gif@gmail.com',

    # Choose your license
    license='MIT',

    # What does your project relate to?
    keywords='BayesianOptimization MachineLearning GaussianProcesses HyperparameterOptimization',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(exclude=['contrib', 'docs', 'tests']),

    # Alternatively, if you want to distribute just a my_module.py, uncomment
    # this:
    #   py_modules=["my_module"],

    # List run-time dependencies here.  These will be installed by pip when
    # your project is installed. For an analysis of "install_requires" vs pip's
    # requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=['numpy'],

    # To provide executable scripts, use entry points in preference to the
    # "scripts" keyword. Entry points provide cross-platform support and allow
    # pip to create the appropriate form of executable for the target platform.
    entry_points={
    },
)
