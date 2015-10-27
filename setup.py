#!/usr/bin/env python
from setuptools import setup

setup(
    name='rcc8',
    version='0.0.1',
    description='Region Connection Calculus lib',
    url='git@github.com:dnsantos/rcc8.git',
    author='Danilo Nunes',
    author_email='nunesdanilo@gmail.com',
    packages=['rcc8'],
    install_requires=[
        'numpy==1.9.3',
    ],
)
