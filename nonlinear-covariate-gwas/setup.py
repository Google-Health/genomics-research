"""Builds the DeepNull package.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
"""

from setuptools import setup, find_packages
import pathlib

here = pathlib.Path(__file__).parent.resolve()

# Get the long description from the README file
long_description = (here / 'README.md').read_text(encoding='utf-8')

setup(
    # To support installation via
    #
    # $ pip install deepnull
    name='deepnull',
    version='0.1.1',  # Keep in sync with __init__.__version__.
    description='Models nonlinear interactions between covariates and phenotypes',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/google-health/genomics-research/tree/main/nonlinear-covariate-gwas',
    author='Google LLC',
    keywords='GWAS',
    packages=['deepnull'],
    package_dir={'deepnull': '.'},
    python_requires='>=3.7, <4',
    install_requires=[],
)
