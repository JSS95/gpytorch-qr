# GPyTorch-QR

[![Supported Python Versions](https://img.shields.io/pypi/pyversions/gpytorch-qr.svg)](https://pypi.python.org/pypi/gpytorch-qr/)
[![PyPI Version](https://img.shields.io/pypi/v/gpytorch-qr.svg)](https://pypi.python.org/pypi/gpytorch-qr/)
[![License](https://img.shields.io/github/license/JSS95/gpytorch-qr)](https://github.com/JSS95/gpytorch-qr/blob/master/LICENSE)
[![CI](https://github.com/JSS95/gpytorch-qr/actions/workflows/ci.yml/badge.svg)](https://github.com/JSS95/gpytorch-qr/actions/workflows/ci.yml)
[![CD](https://github.com/JSS95/gpytorch-qr/actions/workflows/cd.yml/badge.svg)](https://github.com/JSS95/gpytorch-qr/actions/workflows/cd.yml)
[![Docs](https://readthedocs.org/projects/gpytorch-qr/badge/?version=latest)](https://gpytorch-qr.readthedocs.io/en/latest/?badge=latest)

Gaussian process quantile regression using GPyTorch.

## Installation

```
$ pip install gpytorch-qr
```

## Documentation

The manual can be found online:

> https://gpytorch-qr.readthedocs.io

If you want to build the document yourself, get the source code and install with `[doc]` dependency.
Then, go to `doc` directory and build the document:

```
$ pip install .[doc]
$ cd doc
$ make html
```

Document will be generated in `build/html` directory. Open `index.html` to see the central page.

## Developing

### Installation

For development features, you must install the package by `pip install -e .[dev]`.

### Re-building examples

Configure the local git filter (run once after cloning):

```
git config filter.nbstripout.clean "nbstripout --keep-output --keep-metadata-keys 'metadata.language_info'"
git config filter.nbstripout.smudge cat
git config filter.nbstripout.required true
```

Then build the examples:

```
jupyter nbconvert --to notebook --execute --inplace examples/*.ipynb
```
