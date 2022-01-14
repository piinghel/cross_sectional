from struct import pack
from setuptools import setup, find_packages

setup(
    name="cross_sectional",
    version="0.0.1",
    packages=find_packages("cross_sectional"),
    install_requires=["pandas>=1.3.4", "plotly>=5.5.0", "numpy>=1.19.2"],
)