# Script to define dawge_planner as a package as well

from setuptools import find_packages, setup

setup(
    name="dawge_planner",
    packages=find_packages(), # find_packages are not installing any extra packages for now
)