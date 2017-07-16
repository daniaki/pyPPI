from setuptools import setup

import os
import glob
import shutil

# Copy script files
home_folder = os.path.normpath(os.path.join(os.path.expanduser('~'), '.pyppi/'))
print("Copying data files to '{}'".format(home_folder))
os.makedirs(home_folder, exist_ok=True)
for file in glob.glob("data/*.*"):
    shutil.copy(file, home_folder)

os.makedirs(home_folder + '/hprd', exist_ok=True)
for file in glob.glob("data/hprd/*.*"):
    shutil.copy(file, home_folder + '/hprd')

os.makedirs(home_folder + '/networks', exist_ok=True)
for file in glob.glob("data/networks/*.*"):
    shutil.copy(file, home_folder + '/networks')

setup(
    name='pyppi',
    version='0.1',
    description='Predict the post-translation modification/signalling function'
                ' of protein-protein interactions',
    long_description='MIT',
    url='http://github.com/daniaki/pyPPI',
    author='Daniel Esposito',
    author_email='danielce90@gmail.com',
    license=open('LICENSE', 'rt').read(),
    packages=[
        'pyppi',
        'pyppi.base',
        'pyppi.data',
        'pyppi.data_mining',
        'pyPPI.model_selection',
        'pyPPI.models',
        'pyPPI.network_analysis',
    ],
    install_requires=[
        'scikit-learn>=0.18.1',
        'pandas>=0.19.1',
        'numpy>=1.11.2',
        'scipy>=0.18.1',
        'bioservices>=1.4.14',
        'goatools>=0.6.10',
        'biopython>=1.68',
        'python-igraph>=0.7.1.post6',
        'scikit-multilearn>=0.0.5',
        'docopt>=0.6.2'
    ]
)
