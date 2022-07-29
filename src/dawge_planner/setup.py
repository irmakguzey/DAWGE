from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

d = generate_distutils_setup(
    packages=['dawge_planner'],
    package_dir={'': 'scripts'}
)
setup(**d)

# from setuptools import find_packages, setup

# setup(
#     name="dawge_planner",
#     packages=find_packages(), # find_packages are not installing any extra packages for now
# )