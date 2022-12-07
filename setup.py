#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name="src",
    version="1.0.0",
    description="Visual Search",
    author=["tailocbmt", "ngfuong"],
    author_email="",
    url="https://github.com/ngfuong/lightning-hydra-template",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=[
        "pytorch-lightning",
        ],
    packages=find_packages(),
)
