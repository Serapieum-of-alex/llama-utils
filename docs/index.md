# llama-utils
LlamaIndex utility package

[![Deploy MkDocs](https://github.com/Serapieum-of-alex/llama-utils/actions/workflows/github-pages-mkdocs.yml/badge.svg)](https://github.com/Serapieum-of-alex/llama-utils/actions/workflows/github-pages-mkdocs.yml)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/llama-utils)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

![GitHub last commit](https://img.shields.io/github/last-commit/Serapieum-of-alex/llama-utils)
![GitHub forks](https://img.shields.io/github/forks/Serapieum-of-alex/llama-utils)
![GitHub Repo stars](https://img.shields.io/github/stars/serapieum-of-alex/llama-utils)
![Codecov](https://img.shields.io/codecov/c/github/serapieum-of-alex/llama-utils)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/5e3aa4d0acc843d1a91caf33545ecf03)](https://www.codacy.com/gh/Serapieum-of-alex/llama-utils/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Serapieum-of-alex/llama-utils&amp;utm_campaign=Badge_Grade)


Current release info
====================

| Name                                                                                                                 | Downloads                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | Version                                                                                                                                                                                                                     | Platforms                                                                                                                                                                                                                                                                                                                                 |
|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [![Conda Recipe](https://img.shields.io/badge/recipe-llama--utils-green.svg)](https://anaconda.org/conda-forge/llama-utils) | [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/llama-utils.svg)](https://anaconda.org/conda-forge/llama-utils) [![Downloads](https://pepy.tech/badge/llama-utils)](https://pepy.tech/project/llama-utils) [![Downloads](https://pepy.tech/badge/llama-utils/month)](https://pepy.tech/project/llama-utils)  [![Downloads](https://pepy.tech/badge/llama-utils/week)](https://pepy.tech/project/llama-utils)  ![PyPI - Downloads](https://img.shields.io/pypi/dd/llama-utils?color=blue&style=flat-square) | [![Conda Version](https://img.shields.io/conda/vn/conda-forge/llama-utils.svg)](https://anaconda.org/conda-forge/llama-utils) [![PyPI version](https://badge.fury.io/py/llama-utils.svg)](https://badge.fury.io/py/llama-utils) | [![Conda Platforms](https://img.shields.io/conda/pn/conda-forge/llama-utils.svg)](https://anaconda.org/conda-forge/llama-utils) [![Join the chat at https://gitter.im/Hapi-Nile/Hapi](https://badges.gitter.im/Hapi-Nile/Hapi.svg)](https://gitter.im/Hapi-Nile/Hapi?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge) |

llama-utils - Large Language Model Utility Package
=====================================================================
**llama-utils** is a large language model utility package

[Full Documentation](https://serapieum-of-alex.github.io/llama-utils/latest/)
[Conda-forge feedstock](https://github.com/conda-forge/llama-utils-feedstock/)

Main Features
-------------

- llama-index

Package Overview
----------------

```mermaid
graph TB
    Package[llama-utils]
    Package --> SubPackage1[Indexing]
    Package --> SubPackage3[Storage]
    SubPackage1 --> Module1[index_manager.py]
    SubPackage1 --> Module2[custom_index.py]
    SubPackage3 --> Module5[storage.py]
    SubPackage3 --> Module6[config_loader.py]
```

complete overview of the design and architecture [here](/docs/design_architecture_diagrams.md)

Installing llama-utils
===============

Installing `llama-utils` from the `conda-forge` channel can be achieved by:

```
conda install -c conda-forge llama-utils=0.2.0
```

It is possible to list all the versions of `llama-utils` available on your platform with:

```
conda search llama-utils --channel conda-forge
```

## Install from GitHub

to install the last development to time, you can install the library from GitHub

```
pip install git+https://github.com/Serapieum-of-alex/llama-utils
```

## pip

to install the last release, you can easily use pip

```
pip install llama-utils==0.2.0
```

Quick start
===========
- First download ollama from here [ollama](https://ollama.com/download) and install it.
- Then run the following command to pull the `llama3` model
```
ollama pull llama3
```
- Then run ollama server (if you get an error, check the errors section below to solve it)
```
ollama serve
```
Now you can use the `llama-utils` package to interact with the `ollama` server

```python
from llama_utils.retrieval.storage import Storage
STORAGE_DIR= "examples/data/llama3"
storage = Storage.create()
data_path = "examples/data/essay"
docs = storage.read_documents(data_path)
storage.add_documents(docs)
storage.save(STORAGE_DIR)
```
