import unittest
from pyppi.tests import (
    test_database, test_features, test_uniprot,
    test_db_models, test_managers, test_ontology,
    test_database_utilities
)

if __name__ == "__main__":
    loader = unittest.TestLoader()

    tests = loader.discover(start_dir='./', pattern="test_*.py")
    unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_ontology.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_features.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_db_models.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_uniprot.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_managers.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_database.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(
    #     start_dir='./', pattern="test_database_utilities.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(
    #     start_dir='./', pattern="test_datamining_tools.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_generic.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_hprd.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_kegg.py")
    # unittest.TextTestRunner().run(tests)
