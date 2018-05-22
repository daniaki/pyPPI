# Introduction
A computational tool for annotating the edges in a protein-protein interaction 
(PPI) network with post-translational modification (PTM) and functional labels.
The tool serves two main purposes: 

 1. Mining protein-protein interactions labelled with PTMs and higher level 
 signalling function (activation, inhibition etc) from various databases
 such as [KEGG](http://www.genome.jp/kegg/)and [HPRD](http://www.hprd.org/).

 2. Provide machine learning API that can be used to 
  classify the PTM of binary protein interactions encoded by [UniProt](http://www.uniprot.org/)
  accessions to both learn from a training dataset and  and classify new/unseen 
  classify interactions.

This can be used as a command line tool with the provided scripts, or custom
made scripts. Alternatively, the tool can be integrated into an existing
project using the simple API described in the examples. See the documentation
for an in-depth guide.

# Basic install guide
First clone the repository or download it and extract it to a convenient location. To install the package and all requirements:

```python
python setup.py install
```

To download all required data:

```python
python setup.py download_data
```

To test your installation:

```python
python setup.py test 
```
Or
```python
python test.py
```

Before using the API or provided scripts, you will need to build the training
database and train an initial classifier. Once you have installed the package
you can move the script files to any directory of your choice. Once you have
changed your working directory to the script directory:

```python
python build_data.py -h
python build_data.py
```

# Documentation
The documentation is not currently hosted (upcoming). To build a local copy of the documentation `cd` into `docs` and run the make script:

```bash
make html
```

This will build a html version of the documentation in the folder `build`. To view the documentation, open the file `index.html`.
