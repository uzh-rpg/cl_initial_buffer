from setuptools import setup, find_packages

setup(name='initial_buffer',
      version='1.0.0',
      author='Nico & Yunlong',
      author_email='nmessi@ifi.uzh.ch',
      license="BSD-3-Clause",
      packages=find_packages(),
      description='Initial state buffer to speed up training convergence',
      python_requires='>=3.6',
      install_requires=[
            "torch>=1.4.0",
            "torchvision>=0.5.0",
            "numpy>=1.16.4"
      ],
      )
