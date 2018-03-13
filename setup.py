try:
    from setuptools import setup, Command
except:
    from distutils.core import setup, Command

import sys
import os
import glob
import shutil
import importlib

bioservices_cache_linux = ".cache/bioservices/"
bioservices_cache_darwin = "Library/Caches/bioservices/"
bioservices_cache_win = "AppData/Local/bioservices/bioservices/"

PLATFORM = sys.platform
if PLATFORM == 'linux':
    CACHE_INSTALL = os.path.normpath(
        os.path.join(os.path.expanduser('~'), bioservices_cache_linux)
    )
elif PLATFORM == 'darwin':
    CACHE_INSTALL = os.path.normpath(
        os.path.join(os.path.expanduser('~'), bioservices_cache_darwin)
    )
elif PLATFORM == 'win32':
    CACHE_INSTALL = os.path.normpath(
        os.path.join(os.path.expanduser('~'), bioservices_cache_win)
    )
else:
    print("Unsupported platform {}.".format(PLATFORM))
    sys.exit(0)


base = os.path.join(os.path.expanduser('~'), '.pyppi/')
home_folder = os.path.normpath(base)
os.makedirs(home_folder, exist_ok=True)
os.makedirs(home_folder, exist_ok=True)
os.makedirs(home_folder + '/networks', exist_ok=True)
os.makedirs(home_folder + '/networks', exist_ok=True)


def _install_cache():
    print("Copying data files to bioservices cache '{}'".format(CACHE_INSTALL))
    os.makedirs(CACHE_INSTALL, exist_ok=True)
    os.makedirs(CACHE_INSTALL + '/Cache', exist_ok=True)
    for file_ in glob.glob("data/bioservices/*.*"):
        shutil.copy(file_, CACHE_INSTALL)
    for file_ in glob.glob("data/bioservices/Cache/*.*"):
        shutil.copy(file_, CACHE_INSTALL)


def _download_data():
    download_program_data = getattr(
        importlib.import_module("pyppi.base.io"),
        "download_program_data"
    )
    print("Downloading required data. This may take around 15 minutes.")
    download_program_data()
    print("Copying HPRD files to '{}'".format(home_folder))
    for file_ in glob.glob("data/hprd/*.*"):
        shutil.copy(file_, home_folder + '/networks')


def _install_data():
    # Assuming the data folder is pre-downloaded.
    print("Copying data files to '{}'".format(home_folder))
    for file_ in glob.glob("data/*.*"):
        shutil.copy(file_, home_folder)
    for file_ in glob.glob("data/hprd/*.*"):
        shutil.copy(file_, home_folder + '/networks')
    for file_ in glob.glob("data/networks/*.*"):
        shutil.copy(file_, home_folder + '/networks')


class PassThroughCommand(Command):
    """Pass through command to run the above callbacks"""

    user_options = []

    def initialize_options(self):
        """Abstract method that is required to be overwritten"""
        pass

    def finalize_options(self):
        """Abstract method that is required to be overwritten"""
        pass


class DownloadData(PassThroughCommand):
    def run(self):
        _download_data()


class InstallData(PassThroughCommand):
    def run(self):
        _install_data()


class InstallCache(PassThroughCommand):
    def run(self):
        _install_cache()


class RunTests (PassThroughCommand):
    def run(self):
        os.system("python test.py")


setup(
    name='PyPPI',
    version='1.0b0',
    description='Predict the post-translation modification/signalling '
                'function of protein-protein interactions',
    url='http://github.com/daniaki/pyPPI',
    author='Daniel Esposito',
    author_email='danielce90@gmail.com',
    license=open('LICENSE', 'rt').read(),
    cmdclass={
        'download_data': DownloadData,
        'install_data': InstallData,
        'install_cache': InstallCache,
        'test': RunTests
    },
    packages=[
        'pyppi',
        'pyppi.base',
        'pyppi.data_mining',
        'pyppi.database',
        'pyppi.model_selection',
        'pyppi.models',
        'pyppi.network_analysis',
        'pyppi.predict',
        'pyppi.tests'
    ],
    install_requires=[
        'scikit-learn>=0.18.1',
        'pandas>=0.19.1',
        'numpy>=1.11.2',
        'scipy>=0.18.1',
        'bioservices>=1.4.14',
        'biopython>=1.68',
        'matplotlib',
        'docopt',
        'sqlalchemy',
        'joblib'
    ]
)
