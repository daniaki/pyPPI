from setuptools import setup

setup(
    name='pyPPI',
    version='0.1',
    description='Predict the post-translation modification/signalling function'
                ' of protein-protein interactions',
    url='http://github.com/daniaki/pyPPI',
    author='Daniel Esposito',
    author_email='danielce90@gmail.com',
    license='MIT',
    packages=['pyPPI'],
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
    ],
    zip_safe=False
)
