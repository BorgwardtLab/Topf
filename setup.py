from setuptools import find_packages
from setuptools import setup

setup(
    name='topf',
    version='0.1',
    packages=find_packages(),
    install_requires=['numpy'],
    packages=['topf'],

    package_data={
        # Not sure whether it is smart to distribute the logo along with
        # the package, but here we go...
        '': ['topf.svg'],
    },

    author='Bastian Rieck',
    author_email='bastian.rieck@bsse.ethz.ch',
    description='A package for performing topological peak filtering',
    license='MIT',
    keywords='topology tda peak peak-filtering',
    url='https://github.com/BorgwardtLab/topf'
)
