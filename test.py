import unittest
from pyppi.tests import (
    test_database,
    test_database_utilities,
    test_features,
    test_uniprot,
    test_db_models,
    test_validators,
    test_ontology,
    test_datamining_tools,
    test_generic,
    test_hprd,
    test_kegg,
    test_sampling,
    test_chain,
    test_br,
    test_kfold,
    test_predict,
    test_predict_utilities,
    test_base_utilities,
    test_model_utilities
)

if __name__ == "__main__":
    loader = unittest.TestLoader()

    tests = loader.discover(start_dir='./', pattern="test_*.py")
    unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_database.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(
    #     start_dir='./', pattern="test_database_utilities.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_features.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_uniprot.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_db_models.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_validators.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_ontology.py")
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

    # tests = loader.discover(start_dir='./', pattern="test_sampling.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_chain.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_br.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_kfold.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_predict.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(
    #     start_dir='./', pattern="test_predict_utilities.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_base_utilities.py")
    # unittest.TextTestRunner().run(tests)

    # tests = loader.discover(start_dir='./', pattern="test_model_utilities.py")
    # unittest.TextTestRunner().run(tests)
