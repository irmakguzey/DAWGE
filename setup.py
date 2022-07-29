# Script to define contrastive_learning as a package
from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup
from setuptools import find_packages, setup

setup(
    name="contrastive_learning",
    packages=find_packages(), # find_packages are not installing any extra packages for now
)

d = generate_distutils_setup(
    packages=['dawge_planner'],
    package_dir={'': 'src'}
)
setup(**d)