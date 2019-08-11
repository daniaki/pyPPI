import pytest

from ...constants import GeneOntologyCategory
from ..types import GeneOntologyTermData


class TestGeneOntologyTermData:
    def test_raises_value_error_if_category_is_unknown(self):
        with pytest.raises(ValueError):
            GeneOntologyTermData(
                identifier="GO:123456",
                category="Unknown",
                name="A",
                description=None,
                obsolete=False,
            )

    def test_no_value_error_if_category_is_known(self):
        GeneOntologyTermData(
            identifier="GO:123456",
            category=GeneOntologyCategory.biological_process,
            name="A",
            description=None,
            obsolete=False,
        )
