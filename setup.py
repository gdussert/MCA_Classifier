from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='mca_classifier',
    version='0.1.0',    
    description='A Python package',
    packages=['mca_classifier'],
    python_requires='>=3.11',
    install_requires=requirements,
)