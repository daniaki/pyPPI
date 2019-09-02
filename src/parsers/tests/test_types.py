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

    def test_set_drops_duplicates(self):
        e1 = types.InteractionEvidenceData("1", "MI:0001")
        e2 = types.InteractionEvidenceData("1", None)
        e3 = types.InteractionEvidenceData("2", "MI:0001")
        e4 = types.InteractionEvidenceData("1", "MI:0001")

        items = set([e1, e2, e3, e4])
        assert len(items) == 3
        assert e1 in items
        assert e2 in items
        assert e3 in items
        assert e4 in items


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

    def test_set_drops_duplicates(self):
        # Same hash but different __eq__ should result in both being in a set.
        instance_1 = types.InteractionData(
            source="P12345",
            target="P12346",
            organism=9606,
            databases=["kegg"],
            labels=["activation"],
            evidence=[
                types.InteractionEvidenceData("3", "MI:0003"),
                types.InteractionEvidenceData("2", "MI:0002"),
                types.InteractionEvidenceData("1", "MI:0001"),
            ],
        )
        instance_2 = types.InteractionData(
            source="P12345",
            target="P12346",
            organism=1000,
            databases=["kegg"],
            labels=["methylation"],
            evidence=[
                types.InteractionEvidenceData("3", "MI:0003"),
                types.InteractionEvidenceData("2", "MI:0002"),
            ],
        )
        items = set([instance_1, instance_2, instance_1, instance_2])
        assert len(items) == 2
        assert instance_1 in items
        assert instance_2 in items


class TestPfamTermData:
    def test_hash_is_based_on_tuple(self):
        instance = types.PfamTermData("1", "Name", "Description")
        assert hash(instance) == hash(
            (instance.identifier, instance.name, instance.description)
        )

    def test_set_drops_duplicates(self):
        i1 = types.PfamTermData("1", "Name", "Description")
        i2 = types.PfamTermData("1", "Name", "Description")
        i3 = types.PfamTermData("1", "Name", None)
        i4 = types.PfamTermData("1", None, "Description")

        items = set([i1, i2, i3, i4])
        assert len(items) == 3
        assert i1 in items
        assert i2 in items
        assert i3 in items
        assert i4 in items


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

    def test_set_drops_duplicates(self):
        i1 = types.InterproTermData("1", "Name", "Description", "Type")
        i2 = types.InterproTermData("1", "Name", "Description", "Type")
        i3 = types.InterproTermData("1", None, "Description", "Type")
        i4 = types.InterproTermData("1", "Name", "Description", None)

        items = set([i1, i2, i3, i4])
        assert len(items) == 3
        assert i1 in items
        assert i2 in items
        assert i3 in items
        assert i4 in items


class TestKeywordTermData:
    def test_hash_is_based_on_tuple(self):
        instance = types.KeywordTermData("1", "Name", "Description")
        assert hash(instance) == hash(
            (instance.identifier, instance.name, instance.description)
        )

    def test_set_drops_duplicates(self):
        i1 = types.KeywordTermData("1", "Name", "Description")
        i2 = types.KeywordTermData("1", "Name", "Description")
        i3 = types.KeywordTermData("1", None, "Description")
        i4 = types.KeywordTermData("1", "Name", None)

        items = set([i1, i2, i3, i4])
        assert len(items) == 3
        assert i1 in items
        assert i2 in items
        assert i3 in items
        assert i4 in items


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

    def test_set_drops_duplicates(self):
        i1 = types.GeneOntologyTermData(
            identifier="GO:123456",
            category=GeneOntologyCategory.biological_process,
            obsolete=False,
            name="Name",
            description=None,
        )
        i2 = types.GeneOntologyTermData(
            identifier="GO:123456",
            category=GeneOntologyCategory.biological_process,
            obsolete=False,
            name="Name",
            description=None,
        )
        i3 = types.GeneOntologyTermData(
            identifier="GO:123456",
            category=GeneOntologyCategory.cellular_component,
            obsolete=False,
            name="Name",
            description=None,
        )
        i4 = types.GeneOntologyTermData(
            identifier="GO:123456",
            category=GeneOntologyCategory.molecular_function,
            obsolete=False,
            name="Name",
            description=None,
        )

        items = set([i1, i2, i3, i4])
        assert len(items) == 3
        assert i1 in items
        assert i2 in items
        assert i3 in items
        assert i4 in items


class TestGeneData:
    def test_hash_is_based_on_tuple(self):
        instance = types.GeneData("BRCA1", "primary")
        assert hash(instance) == hash((instance.symbol, instance.relation))

    def test_error_invalid_relation(self):
        with pytest.raises(ValueError):
            types.GeneData("BRCA1", "aa")

    def test_set_drops_duplicates(self):
        i1 = types.GeneData("BRCA1", "primary")
        i2 = types.GeneData("BRCA1", "primary")
        i3 = types.GeneData("BRCA1", "synonym")
        i4 = types.GeneData("BRCA3", "synonym")

        items = set([i1, i2, i3, i4])
        assert len(items) == 3
        assert i1 in items
        assert i2 in items
        assert i3 in items
        assert i4 in items

