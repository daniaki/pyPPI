import pytest

from ...constants import GeneOntologyCategory
from .. import types


class TestInteractionEvidenceData:
    def test_hash_is_based_on_tuple(self):
        instance = types.InteractionEvidenceData("1", "MI:0001")
        assert hash(instance) == hash((instance.pubmed, instance.psimi))

    def test_eq_delegate_to_super_if_unknown_type(self):
        instance = types.InteractionEvidenceData("1", "MI:0001")
        assert instance != 1

    def test_equal_compares_based_on_hash(self):
        assert types.InteractionEvidenceData(
            "1", "MI:0001"
        ) != types.InteractionEvidenceData("1", None)

        assert types.InteractionEvidenceData(
            "1", "MI:0001"
        ) == types.InteractionEvidenceData("1", "MI:0001")


class TestInteractionData:
    def test_hash_is_based_on_source_and_target(self):
        instance = types.InteractionData(
            source="P12345", target="P12346", organism=9606
        )
        assert hash(instance) == hash(
            (instance.source, instance.target, instance.organism)
        )

    def test_type_error_adding_to_different_type(self):
        with pytest.raises(TypeError):
            instance = types.InteractionData(
                source="P12345", target="P12346", organism=9606
            )
            instance + []

    def test_value_error_adding_instance_with_different_hash(self):
        instance_1 = types.InteractionData(
            source="P12345", target="P12346", organism=9606
        )
        instance_2 = types.InteractionData(
            source="P12345", target="P12347", organism=9606
        )
        with pytest.raises(ValueError):
            instance_1 + instance_2

    def test_add_sorts_and_removes_duplicate_labels(self):
        instance_1 = types.InteractionData(
            source="P12345", target="P12346", organism=9606, labels=["b", "c"]
        )
        instance_2 = types.InteractionData(
            source="P12345", target="P12346", organism=9606, labels=["a"]
        )
        assert (instance_1 + instance_2).labels == ["a", "b", "c"]

    def test_add_sorts_and_removes_duplicate_databases(self):
        instance_1 = types.InteractionData(
            source="P12345",
            target="P12346",
            organism=9606,
            databases=["b", "c"],
        )
        instance_2 = types.InteractionData(
            source="P12345", target="P12346", organism=9606, databases=["a"]
        )
        assert (instance_1 + instance_2).databases == ["a", "b", "c"]

    def test_add_sorts_and_removes_duplicate_evidence_terms(self):
        instance_1 = types.InteractionData(
            source="P12345",
            target="P12346",
            organism=9606,
            evidence=[
                types.InteractionEvidenceData("3", "MI:0003"),
                types.InteractionEvidenceData("2", "MI:0002"),
                types.InteractionEvidenceData("1", "MI:0001"),
            ],
        )
        instance_2 = types.InteractionData(
            source="P12345",
            target="P12346",
            organism=9606,
            evidence=[types.InteractionEvidenceData("2", "MI:0002")],
        )
        assert (instance_1 + instance_2).evidence == [
            types.InteractionEvidenceData("1", "MI:0001"),
            types.InteractionEvidenceData("2", "MI:0002"),
            types.InteractionEvidenceData("3", "MI:0003"),
        ]


class TestPfamTermData:
    def test_hash_is_based_on_tuple(self):
        instance = types.PfamTermData("1", "Name", "Description")
        assert hash(instance) == hash(
            (instance.identifier, instance.name, instance.description)
        )


class TestInterproTermData:
    def test_hash_is_based_on_tuple(self):
        instance = types.InterproTermData("1", "Name", "Description", "Type")
        assert hash(instance) == hash(
            (
                instance.identifier,
                instance.name,
                instance.description,
                instance.entry_type,
            )
        )


class TestKeywordTermData:
    def test_hash_is_based_on_tuple(self):
        instance = types.KeywordTermData("1", "Name", "Description")
        assert hash(instance) == hash(
            (instance.identifier, instance.name, instance.description)
        )


class TestGeneOntologyTermData:
    def test_hash_is_based_on_tuple(self):
        instance = types.GeneOntologyTermData(
            identifier="GO:123456",
            category=GeneOntologyCategory.biological_process,
            obsolete=False,
            name="Name",
            description=None,
        )
        assert hash(instance) == hash(
            (
                instance.identifier,
                instance.category,
                instance.obsolete,
                instance.name,
                instance.description,
            )
        )

    def test_raises_value_error_if_category_is_unknown(self):
        with pytest.raises(ValueError):
            types.GeneOntologyTermData(
                identifier="GO:123456",
                category="Unknown",
                name="A",
                description=None,
                obsolete=False,
            )
