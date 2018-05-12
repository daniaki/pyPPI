Installation
============
This section will walk you through installation. The build and data installation process is identical for each operating system.


.. attention:: It is **strongly** recommended that you install **PyPPI** in a separate virtual environment to avoid package version issues. Installing **PyPPI** may update some of your existing packages causing problems for any other packages relying on the previous versions to function correctly.

.. attention:: It is **strongly** recommended that Windows users use `Anaconda <https://www.anaconda.com/download/>`_ to handle the installation of requirements. See the Anaconda documentation `here <https://conda.io/docs/index.html>`_ for instructions on how to do this.


Setting your python path
------------------------
If you try to use python in the command prompt or terminal and you see an error along the lines of ``unrecognised command`` it means that your path variable has not yet been configured. To do this on windows, open up a command prompt and use the following command:

.. code-block:: none

    set PATH=<path to your python installation>;%PATH%

Here's an example command assuming you have installed 64bit python from `here <https://www.python.org/downloads/>`_ files directory:

.. code-block:: none

    set PATH=C:\Users\<username>\AppData\Programs\Python\Python36\;%PATH%

For Anaconda users, the path should already be set, but here's the command anyway:

.. code-block:: none

    set PATH=<path to your anaconda installation>;%PATH%

Anaconda by default is installed to ``C:\Users\<username>\Anaconda3``. On MacOS and Linux you can use a similar command:

.. code-block:: none

    export PATH=/path/to/python/installation/:$PATH


Creating/managing virtual environments
--------------------------------------

Anaconda
~~~~~~~~
If you are using Anaconda as your main python installation you can create a virtual environment with all packages copied from your global environment by typing the following commands into a terminal or command prompt:

.. code-block:: none

    conda create -n pyppi python=3.6 anaconda

To activate and deactivate your new environment:

.. code-block:: none

    source activate pyppi (MacOS/Linux)
    activate pyppi (Windows)

.. code-block:: none

    source deactivate pyppi (MacOS/Linux)
    deactivate pyppi (Windows)

If you are not using Anaconda, read the section below for an alternate way of managing environments.

Virtualenvwrapper
~~~~~~~~~~~~~~~~~
To create a virtual environment, you will need to install the **virtualenv** and **virtualenvwrapper** through **pip** by typing the following commands into a terminal or command prompt:

.. code-block:: none

    pip install virtualenv
    pip install virtualenvwrapper

To create a virtual environment named **pyppi**:

.. code-block:: none

    mkvirtualenv pyppi

Whenever you need to use **PyPPI** (including script usage) or install new packages to this environment you will need to activate it:

.. code-block:: none

    workon pyppi

Once finished, to deactivate the environment and return to the global python environment:

.. code-block:: none

    deactivate pyppi


Requirements
------------
PyPPI requires the following packages in order to run:

- scikit-learn >= 0.19.0
- pandas >= 0.19.1
- numpy >= 1.11.2
- scipy >= 0.18.1
- bioservices >= 1.4.14
- biopython >= 1.68
- matplotlib
- docopt
- sqlalchemy
- joblib

The installation of these packages will be handled automatically. However, if choosing to install without Anaconda on Windows, you will need to manually install the following pre-built wheels:

- `Biopython <https://www.lfd.uci.edu/~gohlke/pythonlibs/#biopython>`_
- `Greenlet <https://www.lfd.uci.edu/~gohlke/pythonlibs/#greenlet>`_
- `Gevent <https://www.lfd.uci.edu/~gohlke/pythonlibs/#gevent>`_

If you have problems installing Scipy, Numpy, Pandas, Matplotlib or SciKit-Learn, download and install the pre-built wheel from this `repository <https://www.lfd.uci.edu/~gohlke/pythonlibs/>`_. To install a wheel, open a terminal/command prompt session and change directory containing your downloaded wheels (``cd <directory name>``). To install a wheel, enter the following command:

.. code-block:: none

    pip install <downloaded file ending in *.whl>

If you are going to use Anaconda to install these dependencies manually, you may need to add these channels by using the following commands:

.. code-block:: none

    conda config --add channels conda-forge
    conda config --add channels bioconda


Building
--------
To build and install this package and all it's dependencies, download or clone the repository to your home directory. Using either the terminal or command prompt:

.. code-block:: none

    python setup.py install


Downloading data
----------------
**PyPPI** requires several data files to be downloaded before the package can be used. To download these files in to the home directory which is ``C:\Users\<username>\.pyppi\`` for Windows users and ``~\.pyppi\`` for MacOS/Linux users:

.. code-block:: none

    python setup.py download_data

Alternatively, if you have already downloaded the packaged data provided by us, then copy this data into the **data** directory in the downloaded/cloned github repository. After you have done this, run the following command:

.. code-block:: none

    python setup.py install_data


Installing cache data (optional)
--------------------------------
If you would like to install the **bioservices** cache that was used to during publication, although there is no guarantee that **bioservices** will use this cache as intended, and may create a new one.

.. code-block:: none

    python setup.py install_cache


Testing your installing
-----------------------
Once you have completed the above steps, to make sure the installation is correct you should run the provided test suite:

.. code-block:: none

    python setup.py test

Note that some of these tests will require a working internet connection to test database access.


Building the database
---------------------
**PyPPI** needs to build the initial database before you are able to run the provided scripts or use the API in your own workflows. To do this cd in the scripts directory in the github repository. Alternatively, you may copy these scripts to any location you wish and delete the original repository. Open a command prompt or terminal, activate your virtual environment containing the installation if you have one and run the following command:

.. code-block:: none

    python build_data.py

There are three additional parameters which you can pass into the script

- ``clear_cache``: This will clear the **bioservices** cache. Do this if you want to ensure you get the latest information from **KEGG** and **UniProt**.
- ``n_jobs``: The number of processes to use when pre-computing features for the parsed interactions. This will default to 1 process.
- ``verbose``: Supply this if you would intermediate messages printed to your console.

For a completely fresh build run:

.. code-block:: none

    python build_data.py --clear_cache --verbose --n_jobs=<# of processes>

Alternatively you can run the following for more help:

.. code-block:: none

    python build_data.py --help

