from setuptools import setup, find_packages
import os

with open('requirement.txt') as fp:
        install_requires = fp.read()

setup(name='PHME-data-contest',
      version='0.1',
      install_requires=install_requires,
      description='Implentation of the results on the public training dataset for PHME-data-contest-2022',
      long_description=open(os.path.join(os.path.dirname(__file__), 'README.md')).read(),
      url='https://github.vitesco.io/alexandregft/PHME-data-contest',
      author_email='alexandre.gaffet@orange.fr',
      license='MIT',
      packages=find_packages(),
      zip_safe=False)
