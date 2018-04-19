"""A setuptools based setup module.
See:
https://packaging.python.org/tutorials/distributing-packages/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

VERSION = '0.0.1'

setup(
    name='lolpy',
    version=VERSION,
    description='Linear Optimal Low Rank Projection',
    url='https://github.com/j1c/lol',
    author='Jaewon Chung',
    author_email='j1c@jhu.edu',
    license='GPL',
    keywords='dimensionality reduction',
    packages=['lol'],  # Required
    install_requires=['scipy>=1.0.0', 'scikit-learn==0.19.1', 'numpy>=1.14.2'],
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Visualization',
        'License :: OSI Approved :: GPL License',
        'Programming Language :: Python :: 3.6',
    ])
