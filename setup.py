"""This module contains the packaging routine for oddsgym"""
from os import path
from setuptools import setup, find_packages
from configparser import ConfigParser


requirements = {'install_requires': ['gym', 'more-itertools', 'pandas']}
requirements['extra_requires'] = {'all': requirements['install_requires'] + ['tensorflow<2', 'stable-baselines']}
with open(path.join(path.abspath(path.dirname(__file__)), 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


def _get_version():
    with open(path.join(path.dirname(__file__), 'setup.cfg')) as setup_configuration_file:
        parser = ConfigParser()
        parser.read_string(setup_configuration_file.read())
        return parser['bumpversion']['current_version']


setup(packages=find_packages(exclude=["tests*"]),
      install_requires=requirements['install_requires'],
      version=_get_version(),
      python_requires=">=3.6",
      long_description=long_description,
      long_description_content_type='text/markdown')
