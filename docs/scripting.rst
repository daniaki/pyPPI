Using the scripts
=================
.. attention:: The use of the following scripts assume you have built the initial database as instructed in **Installation**.

Validation
----------
This script is to be used to compute the cross-validation performance on the training data for a given configuration. To see a detailed description of each parameter you can provide the script use the following:

.. code-block:: none

    python validation.py --help

This script will assume that you have a directory called results in the same folder as this script. If not you will need to create this folder before proceeding. This script generally a minimum of 2 hours to run on a 16-thread machine using all threads when using all features (**InterPro**, **Pfam** and **Gene Ontology**).


Prediction
----------
Use this script if you would like to make a prediction on the interactome or your own custom input file. To see a detailed list of script parameters:

.. code-block:: none

    python predict.py --help

This script imposes a few constraints. If using the pre-trained classifier you will not need to specify which features to use. If you wish to retrain a classifier with ``--retrain``, then you will also need to supply a selection. If you supply a selection without ``--retrain`` that does not match a previously saved classifier, then you receive an error.

By default, this script assumes that a directory **results** exists. You will need to create it before continuing or supply the **absolute** path to your own custom output directory.

Interactome
~~~~~~~~~~~
To make predictions on the interactome interactions built during installation, do not supply any input to the script.

Custom input
~~~~~~~~~~~~
To make predictions on your own protein-protein interactions, you will need to supply the **absolute** file path to a **tab separated** edge list of **uppercase** **UniProt** accession identifiers. An example input file is shown below:

.. code-block:: none

    source  target
    P00123  Q12345
    P70123  Q02395

You must include the header exactly as above. Some things to note:

- Duplicate entries will be removed.
- Supplied UniProt accessions will be mapped to the latest UniProt accessions.
- Interactions with invalid UniProt accessions will not be used. You can view these in the output file.
- If a supplied UniProt accession maps to two valid accessions, this will become two new interactions. For example, if (A, D) maps to B and C, then two interactions will appear in the output: (B, D) and (C, D).
- The source and target in each row will be sorted in ascending alphabetical order.

Output explanation
~~~~~~~~~~~~~~~~~~
Once completed, the following files will be saved to your chosen output directory:

- **classifier.pkl**: A pickled tuple containing the trained classifier, the feature selection it was trained on and a `MultiLabelBinarizer <http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MultiLabelBinarizer.html>`_ which was used during training to transform training labels into a binary indicator matrix for multi-label prediction.
- **dataset_statistics.json**: A json file containing various bits of information about the training and supplied dataset for your own records.
- **prediction_distribution.json**: A json file containing the number of instances for each label with a prediction probability of 0.5 or greater.
- **predictions.tsv**: Tab separated edge-list of predicted probabilities for each label. See below for more details.
- **settings.json**: The settings the script was run with.
- **threshold.png**: Threshold curve for the number of interactions that could be assigned at least one label at a given probability.
- **thresholds.csv**: The data used to compute the above curve.

The prediction file
~~~~~~~~~~~~~~~~~~~
Below are the header columns in the `predictions.tsv` file containing your annotated network, and their explanations:

- **protein_a**: The most recent UniProt accesion corresponding the first protein in the interaction.
- **protein_b**: The most recent UniProt accesion corresponding the second protein in the interaction.
- **gene_a**: The gene name for ``protein_a``.
- **gene_b**: The gene name for ``protein_b``.
- **input_source**: The original supplied UniProt accession for ``protein_a``.
- **input_target**: The original supplied UniProt accession for ``protein_b``.
- **<labelname>-pr**: The prediction probability for this label for a particular row.
- **sum-pr**: The sum of all of the prediction probabilities for the row.
- **max-pr**: The max probability observed over all label predictions.
- **classification**: The multi-label classification (comma-separated), computed by considering all labels with a probability of at least 0.5.
- **classification_at_max**: The label with the highest probability, **but not necessarily above 0.5**.
- **proportion_go_used**: The proportion of the GO annotations that could be used during classification (only those seen during training).
- **proportion_interpro_used**: The proportion of the InterPro annotations that could be used during classification (only those seen during training).
- **proportion_pfam_used**: The proportion of the Pfam annotations that could be used during classification (only those seen during training).
- **pubmed**: Pubmed identifiers (comma-separated) supporting this interaction if any could be found in the database.
- **experiment_type**: The Psi-mi accessions supporting a Pubmed identifier if any could be found in the database. Psi-mi identifiers for a particular Pubmed id will be grouped by the character ``'|'``. Groups will be comma-separated. If no accessions exist for an identifier then the value will be the string ``'None'``.


Inducing subnetworks
--------------------
Given a ``predictions.tsv`` file, you will be able to induce a subnetwork corresponding to a particular pathway or label. As before, to see more information regarding script parameters:

.. code-block:: none

    python induce.py --help


The output from this script will be two files: a ``.noa`` file, containing attributes for each node in the induced network, and the network itself in the ``.tsv`` file. You can then load these networks into `Cytoscape <`http://www.cytoscape.org/>`_ for further analysis.

From a pathway
~~~~~~~~~~~~~~
You can supply a list of Gene UniProt identifiers and the script will return all itneractions containing that identifer as either a source or target given that at least one label has been predicted above or equal to the probability threshold defined by ``--threshold``.

From a label
~~~~~~~~~~~~
You can supply a label name and the script will return all interactions that have prediction probability for that label above or equal to the probability threshold defined by ``--threshold``.





