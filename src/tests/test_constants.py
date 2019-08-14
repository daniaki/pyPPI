import pytest

from .. import constants


class TestGeneOntologyCategory:
    def test_converts_single_letter_to_category(self):
        assert (
            constants.GeneOntologyCategory.letter_to_category("C")
            == constants.GeneOntologyCategory.cellular_component
        )
        assert (
            constants.GeneOntologyCategory.letter_to_category("F")
            == constants.GeneOntologyCategory.molecular_function
        )
        assert (
            constants.GeneOntologyCategory.letter_to_category("P")
            == constants.GeneOntologyCategory.biological_process
        )

    def test_raises_value_error_when_converting_unknown_single_letter(self):
        with pytest.raises(ValueError):
            constants.GeneOntologyCategory.letter_to_category("?")

    def test_list_returns_list_of_strings(self):
        assert constants.GeneOntologyCategory.list() == [
            constants.GeneOntologyCategory.molecular_function,
            constants.GeneOntologyCategory.biological_process,
            constants.GeneOntologyCategory.cellular_component,
        ]

    def test_choices_returns_list_of_tuples(self):
        assert constants.GeneOntologyCategory.choices() == [
            (
                constants.GeneOntologyCategory.molecular_function,
                constants.GeneOntologyCategory.molecular_function,
            ),
            (
                constants.GeneOntologyCategory.biological_process,
                constants.GeneOntologyCategory.biological_process,
            ),
            (
                constants.GeneOntologyCategory.cellular_component,
                constants.GeneOntologyCategory.cellular_component,
            ),
        ]
