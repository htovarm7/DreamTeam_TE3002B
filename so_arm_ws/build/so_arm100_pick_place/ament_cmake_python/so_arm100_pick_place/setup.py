from setuptools import find_packages
from setuptools import setup

setup(
    name='so_arm100_pick_place',
    version='0.1.0',
    packages=find_packages(
        include=('so_arm100_pick_place', 'so_arm100_pick_place.*')),
)
