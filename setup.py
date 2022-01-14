from struct import pack
from setuptools import setup, find_packages

setup(
    name="cs_trade",
    version="0.0.8",
    packages=find_packages(),
    setup_requires=['wheel'],
    install_requires=["pandas>=1.3.4", "plotly>=5.5.0", "numpy>=1.19.2"],
)