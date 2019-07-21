# -*- coding: utf-8 -*-
#
# Some basic setup for now
#
from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name = "primus",
    version = "0.0.1",
    author = "Tao Hu",
    author_email = "taohu88@gmail.com",
    description = ("prims, a collection of useful transformers especially for pandas"),
    license = license,
    keywords = "machine learning feature engineering",
    url = "https://github.com/taohu88/primus",
    packages=find_packages(),
    long_description=readme,
    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.7',
        "Topic :: App framework",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires='>=3.6',
    install_requires=[
        'dataclasses;python_version>="3.6"',
        'category_encoders>=2.0',
        'pandas>=0.24',
        'numpy>=1.16'
    ],)