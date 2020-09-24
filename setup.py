#nsml: nsml/ml:cuda9.0-cudnn7-tf-1.11torch0.4keras2.2
from distutils.core import setup
import setuptools

print('setup_test.py is running...')

setup(name='PNS_DENSE_WOOK',
      version='1.0',
      install_requires=[
            'opencv-python', 
            'monai',
            'tensorboard',
            'tb-nightly',
            'future']
      ) ## install libraries, 'keras==xx.xx'