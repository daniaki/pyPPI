from setuptools import setup

setup(
    name='pyPPI',
    version='0.1',
    description='Predict the post-translation modification/signalling function'
                ' of protein-protein interactions',
    long_description='MIT',
    url='http://github.com/daniaki/pyPPI',
    author='Daniel Esposito',
    author_email='danielce90@gmail.com',
    license='LICENSE.txt',
    packages=[
        'pyPPI',
        'pyPPI.base',
        'pyPPI.data',
        'pyPPI.data.hprd',
        'pyPPI.data.networks',
        'pyPPI.data_mining',
        'pyPPI.model_selection',
        'pyPPI.models',
        'pyPPI.network_analysis',
    ],
    package_data={
        'pyPPI': [
            'data/*.pkl',
            'data/*.json',
            'data/*.obo',
            'data/*.dat',
            'data/*.list',
            'data/*.tsv',
            'data/*.txt',
            'data/*.csv',
            'data/networks/*.tsv',
            'data/networks/*.mitab',
            'data/networks/*.sif',
            'data/networks/*.txt',
            'data/networks/*.csv',
            'data/hprd/*.txt'
        ]
    },
    scripts=[
        './scripts/build_data.py',
        'scripts/induce_subnetwork.py',
        'scripts/predict.py',
        'scripts/validation.py'
    ],
    install_requires=[
        'scikit-learn',
        'pandas',
        'numpy',
        'scipy',
        'bioservices',
        'goatools',
        'biopython',
        'python-igraph',
        'scikit-multilearn'
    ]
)
