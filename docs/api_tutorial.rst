Using the API
=============
This section will guide you through several ways in which you can interact and use **PyPPI** in your own scripts.

Making predictions
------------------
You can programmatically make predictions by importing :func:`classify_interactions`:

.. code-block:: python

    from pyppi.predict import classify_interactions

    ppis = [('P00533', 'P01133')]
    y_pred, invalid, mapping, labels = classify_interactions(ppis, proba=True, taxon_id=9606)

You may either supply your own classifier or leave the default as ``None``. If you do not supply your own classifier and specify a feature selection to train on, the default classifier and selection that was trained during the build data phase of the installation will be used. If the interaction does not exist, an attempt will be made to create it by downloading the appropriate records from UniProt and creating new :class:`Protein` and :class:`Interaction` instances in the database for the new records and interaction. If you have supplied invalid UniProt accessions or accessions not matching ``taxon_id`` then the interaction will be added to the ``invalid`` list.

The ``mapping`` variable contains a mapping from updated UniProt accessions to supplied accessions in the case that obsoleted or deleted accessions have been supplied. The variable ``labels`` contains a list of label names corresponding to the columns in the prediction matrix ``y_true`` See the API documentation for more details.

Alteratively, you may also supply a list of :class:`Interaction` instances queried from the database instead of a list of accession tuples.


Training a classifier
---------------------
To train the default classifier as published:

.. code-block:: python

    from pyppi.predict.utilities import train_paper_model, DEFAULT_SELECTION

    clf, selection, binariser = train_paper_model(
        rcv_splits=3, # Number of cross-validation splits
        rcv_iter=60, # Number of hyper-parameter search iterations
        scoring='f1', # Scoring to choose best model
        n_jobs_model=1, # Number of processes for the Scikit-Learn model
        n_jobs_br=1, # Number of processes for the Binary Relevance classifier
        n_jobs_gs=-1, # Number of processes for the Grid Search Classifier
        random_state=None, # Random state for classifiers and data sampling
        taxon_id=9606, # Taxonmy ID to select training data from
        verbose=True,  # Log messages to console
        selection= # Default features to train on ['pfam', 'interpro', 'go_mf', 'go_cc', 'go_bp']
    )


You can save this result to your home directory or a custom directory by using the :func:`base.io.save_classifier` function:

.. code-block:: python

    from pyppi.base.io import save_classifier

    home = None
    other = "path/to/directory/"
    save_classifier((clf, selection, binariser), path=home)
    save_classifier((clf, selection, binariser), path=other)


Querying the database
---------------------
This package builds a database with several tables:

- protein
- interaction
- pubmed
- psimi
- reference

These tables can be queried using the `SQLAlchemy <http://docs.sqlalchemy.org/en/latest/>`_ object relational mapper (ORM). For example if you want all proteins that are Human (9606):

.. code-block:: python

    from pyppi.database.models import Protein

    query = Protein.query.filter(Protein.taxon_id == 9606)
    result_as_list = query.all()

    # Get a protein by uniprot_id and print some of it's stored fields.
    protein = Protein.get_by_uniprot_id('P00533')
    print(protein.reviewed)
    print(protein.gene_id)
    print(protein.interpro)

    # The same operation for bulk queries (generally faster).
    query = Protein.query.filter(Protein.uniprot_id.in_(['P00533']))
    result_as_list = query.all()

The variable ``query`` is a `Query <http://docs.sqlalchemy.org/en/latest/orm/query.html>`_ instance, and supports behaviours such as filtering by columns, and other table operations. Similar operations can be done if you would like to filter for specific interactions.

.. code-block:: python

    from pyppi.database.models import Interaction

    # Integer primary key of the Protein since using a query this way
    # Will only search for foreign key matches.
    query = Interaction.query.filter(Interaction.source == 1)

    # Alternatively, get an interaction by it's accessions.
    # Direction is ignored.
    instance = Interaction.get_by_interactors('P00533', 'P01133')

    # Get all interactions containing the label 'Methylation'.
    result = Interaction.get_by_label('Methylation')
    if result is not None:
        for result in result.all():
            print(result.labels_as_list)
            print(result.reference) # A pmid with it's associated PSI-MI
            print(result.pmids) # Supporting publications
            print(result.experiment_types) # Assay detection method

If you have modified any of the fields of a database instance, you can save changes to the database by calling the save method with ``commit=True``.

.. code-block:: python

    instance.save(session=None, commit=True)

Leave ``session`` as ``None`` to save to the default home directory database. If you need to perform save of many instances it is **much** more efficient to import the database session and perform a bulk commit.

.. code-block:: python

    from pyppi.database import db_session

    try:
        db_session.add_all(instances_to_save)
        db_session.commit()
    except:
        db_session.rollback()

The try/except block will rollback any changes made to the database if an error occurs. These are just a few operations that can be done. See the API for these classes for additional details.


Parsing UniProt
---------------
If you have ``dat`` file downloaded from UniProt that you would like to parse into the database from, you can do this using the function :func:`proteins_from_dat`.

.. code-block:: python

    from pyppi.database.utilities import proteins_from_dat

    proteins_from_dat(path_to_file, verbose=True)

This operation will override the fields in any existing proteins with the information parsed from this file. This function allows you to parse proteins from any organism. You may also supply a **gzipped** file to save disk space and memory.


Creating Interactions
---------------------
Finally, if you have a list of UniProt accession tuples representing edges, then you can create new interactions using the utility function :func:`get_or_create_interactions`.

.. code-block:: python

    from pyppi.database.utilities import proteins_from_dat

    ppis = [('P00533', 'P01133')]
    interactions, invalid, mapping = get_or_create_interactions(
        ppis, session=None, taxon_id=9606, verbose=True, n_jobs=1
    )

If any UniProt accessions can not be found during this call, an attempt will be made to download a record mactching the supplied ``taxon_id``. Setting ``n_jobs`` higher than one will speed up the download and feature computation processes. The ``mapping``, and ``invalid`` variables returned are the same as those described in the `Making predictions`_ section.

For more information on these functions and other available utility functions, see the API documentation section.