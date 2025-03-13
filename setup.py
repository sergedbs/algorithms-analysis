from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="algorithm-analysis",
    version="0.1.0",
    description="Empirical analysis of algorithms using Python and Jupyter notebooks",
    author="sergedb",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=requirements,
    python_requires=">=3.7",
)