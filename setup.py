# Copyright 2016 Jason Heeris, jason.heeris@gmail.com
# 
# This file is part of the detan, the deterministic annealing library, and is
# licensed under the 3-clause BSD license distributed with this software.

from setuptools import setup, find_packages

setup(
    name = "detan",
    version = "1.0",
    packages = find_packages(),
        
    install_requires = [
        'numpy',
        'nose',
    ],
)
