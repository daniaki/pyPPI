"""
This module contains submodules used to mine Interaction data from `Kegg`,
`Hprd`, `InnateDB`, `BioPlex` and `Pina2` along with other data processing 
modules. 

The `features` module contains a function to compute the features based on 
two `:class:`.database.models.Protein`s. The `uniprot` module contains several 
functions that use the  `UniProt` class from `bioservices` and http utilities 
of `BioPython` to download and parse `UniProt` records into 
`:class:`.database.models.Protein` instances. The `tools` module contains 
utility functions to turn parsed files into a dataframe of interactions, along 
with several utility functions that operate on those dataframe to perform 
tasks such as removing invalid rows, null rows, merging rows etc. The 
`ontology` module contains methods to parse a `Gene Ontology` obo file into
a dictionary of entries. The `psimi` module does that same, except with
`Psi-Mi` obo files.
"""


__all__ = [
    "features",
    "generic",
    "hprd",
    "kegg",
    "ontology",
    "psimi",
    "tools",
    "uniprot"
]
