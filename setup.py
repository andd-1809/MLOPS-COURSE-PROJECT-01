from setuptools import setup, find_packages

with(open("requirements.txt", "r") as fh):
    requirements = fh.read().splitlines()

setup(
    name="MLApp-MLOPS",
    version="0.1",
    author="AnDD",
    packages=find_packages(),
    install_requires=requirements
)