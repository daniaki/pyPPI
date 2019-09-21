from typing import List, Optional, Tuple, Iterable, Union
from functools import reduce

import pandas as pd
import peewee
import playhouse.fields

from ...settings import DATABASE
from .base import BaseModel, ForeignKeyConstraint
from .identifiers import PsimiIdentifier, PubmedIdentifier, UniprotIdentifier
from .protein import Protein


__all__ = [
    "InteractionDatabase",
    "InteractionLabel",
    "InteractionEvidence",
    "InteractionPrediction",
    "Interaction",
    "ClassifierModel",
]


class InteractionDatabase(BaseModel):
    name = peewee.CharField(
        null=False,
        default=None,
        unique=True,
        max_length=16,
        help_text="Interaction database name.",
    )

    def __str__(self):
        return str(self.name)

    def format_for_save(self):
        self.name = self.name.strip().lower()
        return super().format_for_save()


class InteractionLabel(BaseModel):
    text = peewee.CharField(
        null=False,
        default=None,
        unique=True,
        max_length=64,
        help_text="Interaction type labels.",
    )

    def __str__(self):
        return str(self.text)

    def format_for_save(self):
        self.text = self.text.strip().lower()
        return super().format_for_save()


class InteractionEvidence(BaseModel):
    pubmed = peewee.ForeignKeyField(
        model=PubmedIdentifier,
        null=False,
        default=None,
        backref="evidence_set",
        on_delete=ForeignKeyConstraint.RESTRICT,
        help_text="Pubmed identifier supporting an interaction.",
    )
    psimi = peewee.ForeignKeyField(
        model=PsimiIdentifier,
        null=True,
        default=None,
        backref="evidence_set",
        on_delete=ForeignKeyConstraint.RESTRICT,
        help_text=(
            "PSIMI identifier relating to study performed or interaction type."
        ),
    )

    class Meta:
        indexes = (
            # create a unique on pubmed/psimi
            (("pubmed_id", "psimi_id"), True),
        )

    def __str__(self):
        try:
            return "{}|{}".format(self.pubmed, self.psimi)
        except peewee.DoesNotExist:
            return str(None)

    @classmethod
    def filter_by_pubmed_and_psimi(
        cls,
        identifiers: Iterable[Tuple[str, Optional[str]]],
        query: Optional[peewee.ModelSelect] = None,
    ) -> peewee.ModelSelect:
        """
        Filter evidence table for tuples with matching pubmed and psimi 
        identifiers. Identifiers will be formatted prior to filtering.
        
        Parameters
        ----------
        identifiers : Iterable[Tuple[str, str]]
            String tuples using a (pubmed, psimi) format.
        query : Optional[ModelSelect], optional
            Query to filter. Defaults to all rows.
        
        Returns
        -------
        peewee.ModelSelect
        """
        query = query or cls.all()
        full_queries = []
        null_queries = []
        for (pmid, psimi) in identifiers:
            if psimi is None:
                # Create single column queries if psimi is None.
                null_queries.append(
                    (
                        peewee.fn.Upper(PubmedIdentifier.identifier)
                        == PubmedIdentifier.format(pmid)
                    )
                )
            else:
                # Create queries to search for each tuple using an AND query.
                full_queries.append(
                    (
                        peewee.fn.Upper(PubmedIdentifier.identifier)
                        == PubmedIdentifier.format(pmid)
                    )
                    & (
                        peewee.fn.Upper(PsimiIdentifier.identifier)
                        == PsimiIdentifier.format(psimi)
                    )
                )

        # Search all tuple element using an OR query.
        result_full = None
        result_null = None
        if full_queries:
            result_full = (
                query.select(cls, PubmedIdentifier, PsimiIdentifier)
                .join(PubmedIdentifier, on=(cls.pubmed == PubmedIdentifier.id))
                .switch(cls)
                .join(PsimiIdentifier, on=(cls.psimi == PsimiIdentifier.id))
                .where(
                    reduce(
                        lambda x, y: x | y, full_queries[1:], full_queries[0]
                    )
                )
                .switch(cls)
                .select(cls)
            )
        if null_queries:
            result_null = (
                query.select(cls, PubmedIdentifier)
                .join(PubmedIdentifier, on=(cls.pubmed == PubmedIdentifier.id))
                .switch(cls)
                .select(cls, PubmedIdentifier)
                .where(
                    reduce(
                        lambda x, y: x | y, null_queries[1:], null_queries[0]
                    )
                )
                .switch(cls)
                .select(cls)
            )

        if result_full and result_null:
            return result_full | result_null
        elif result_full:
            return result_full
        elif result_null:
            return result_null
        return cls.none()


class Interaction(BaseModel):
    source = peewee.ForeignKeyField(
        model=Protein,
        null=False,
        default=None,
        backref="interactions_as_source",
        on_delete=ForeignKeyConstraint.RESTRICT,
        help_text="Source protein.",
    )
    target = peewee.ForeignKeyField(
        model=Protein,
        null=False,
        default=None,
        backref="interactions_as_target",
        on_delete=ForeignKeyConstraint.RESTRICT,
        help_text="Target protein.",
    )

    # Interaction metadata.
    labels = peewee.ManyToManyField(
        model=InteractionLabel, backref="interactions"
    )
    evidence = peewee.ManyToManyField(
        model=InteractionEvidence, backref="interactions"
    )
    databases = peewee.ManyToManyField(
        model=InteractionDatabase, backref="interactions"
    )

    class Meta:
        indexes = (
            # create a unique on source/target
            (("source_id", "target_id"), True),
        )

    def __str__(self):
        return str(self.compact)

    @property
    def compact(self) -> Tuple[Optional[str], Optional[str]]:
        """Return source, target string tuple"""
        try:
            source: Optional[str] = str(self.source)
        except peewee.DoesNotExist:
            source = None

        try:
            target: Optional[str] = str(self.target)
        except peewee.DoesNotExist:
            target = None

        return (source, target)

    @classmethod
    def filter_by_pmid(
        cls, pmids: Iterable[str], query: Optional[peewee.ModelSelect] = None
    ) -> peewee.ModelSelect:
        """
        Filter interactions by PubMed reference.
        
        Parameters
        ----------
        pmids : Iterable[str]
            An iterable of PubMed identifiers.
        
        query : Optional[peewee.ModelSelect], optional
            An existing ModelSelect query, by default None and applies
            filter over all rows.
        
        Returns
        -------
        peewee.ModelSelect
        """
        query = query or cls.all()
        pmids = set(PubmedIdentifier.format(pmid).lower() for pmid in pmids)
        return (
            query.select(cls, InteractionEvidence, PubmedIdentifier)
            .join(cls.evidence.get_through_model())
            .join(InteractionEvidence)
            .join(PubmedIdentifier)
            .where(peewee.fn.Lower(PubmedIdentifier.identifier).in_(pmids))
            .switch(cls)
        )

    @classmethod
    def filter_by_database(
        cls,
        databases: Iterable[str],
        query: Optional[peewee.ModelSelect] = None,
    ) -> peewee.ModelSelect:
        """
        Filter interactions by database name.
        
        Parameters
        ----------
        databases : Iterable[str]
            An iterable of database names.
        
        query : Optional[peewee.ModelSelect], optional
            An existing ModelSelect query, by default None and applies
            filter over all rows.
        
        Returns
        -------
        peewee.ModelSelect
        """
        query = query or cls.all()
        dbs = set(db.lower() for db in databases)
        return (
            query.select(cls, InteractionDatabase)
            .join(cls.databases.get_through_model())
            .join(InteractionDatabase)
            .where(peewee.fn.Lower(InteractionDatabase.name).in_(dbs))
            .switch(cls)
        )

    @classmethod
    def filter_by_label(
        cls, labels: Iterable[str], query: Optional[peewee.ModelSelect] = None
    ) -> peewee.ModelSelect:
        """
        Filter interactions by labels.
        
        Parameters
        ----------
        labels : Iterable[str]
            An iterable of labels.
        
        query : Optional[peewee.ModelSelect], optional
            An existing ModelSelect query, by default None and applies
            filter over all rows.
        
        Returns
        -------
        peewee.ModelSelect
        """
        labels = set(l.lower() for l in labels)
        query = query or cls.all()
        return (
            query.select(cls, InteractionLabel)
            .join(cls.labels.get_through_model())
            .join(InteractionLabel)
            .where(peewee.fn.Lower(InteractionLabel.text).in_(labels))
            .switch(cls)
        )

    @classmethod
    def filter_by_source(
        cls,
        identifiers: Iterable[str],
        query: Optional[peewee.ModelSelect] = None,
    ) -> peewee.ModelSelect:
        """
        Filter interactions by UniProt accession appearing as a source node.
        
        Parameters
        ----------
        identifiers : Iterable[str]
            An iterable of accessions.
        
        query : Optional[peewee.ModelSelect], optional
            An existing ModelSelect query, by default None and applies
            filter over all rows.
        
        Returns
        -------
        peewee.ModelSelect
        """
        identifiers = set(i.lower() for i in identifiers)
        fn = peewee.fn.Lower(UniprotIdentifier.identifier)
        query = query or cls.all()
        return (
            query.select(cls, Protein, UniprotIdentifier)
            .join(Protein, on=(cls.source == Protein.id))
            .join(
                UniprotIdentifier,
                on=(Protein.identifier == UniprotIdentifier.id),
            )
            .where(fn.in_(identifiers))
            .switch(cls)
        )

    @classmethod
    def filter_by_target(
        cls,
        identifiers: Iterable[str],
        query: Optional[peewee.ModelSelect] = None,
    ) -> peewee.ModelSelect:
        """
        Filter interactions by UniProt accession appearing as a target node.
        
        Parameters
        ----------
        identifiers : Iterable[str]
            An iterable of accessions.
        
        query : Optional[peewee.ModelSelect], optional
            An existing ModelSelect query, by default None and applies
            filter over all rows.
        
        Returns
        -------
        peewee.ModelSelect
        """
        identifiers = set(i.lower() for i in identifiers)
        fn = peewee.fn.Lower(UniprotIdentifier.identifier)
        query = query or cls.all()
        return (
            query.select(cls, Protein, UniprotIdentifier)
            .join(Protein, on=(cls.target == Protein.id))
            .join(
                UniprotIdentifier,
                on=(Protein.identifier == UniprotIdentifier.id),
            )
            .where(fn.in_(identifiers))
            .switch(cls)
        )

    # @classmethod
    # def filter_by_edge(
    #     cls,
    #     edges: List[Tuple[str, str]],
    #     query: Optional[peewee.ModelSelect] = None,
    # ) -> peewee.ModelSelect:
    #     cls.bulk_create
    #     """
    #     Filter by source and target node identifiers. Search is case
    #     insensitive.

    #     Parameters
    #     ----------
    #     edges : List[Tuple[str, str]]
    #         Uniprot identifiers using a (`source`, `target`) format.

    #     query : Optional[ModelSelect], optional
    #         Query to filter. Defaults to all rows.

    #     Returns
    #     -------
    #     peewee.ModelSelect
    #     """
    #     indices = set(str(edge) for edge in edges)
    #     return cls.select(cls).where(peewee.fn.Lower(cls.index).in_(indices))

    @classmethod
    def to_dataframe(
        cls, query: Optional[peewee.ModelSelect] = None
    ) -> pd.DataFrame:
        if query is None:
            query = cls.select()

        interactions: List[dict] = []
        interaction: Interaction
        index: List[str] = []

        # df column, m2m attribute name
        features = [
            ("go_mf", "go_mf"),
            ("go_bp", "go_bp"),
            ("go_cc", "go_cc"),
            ("keywords", "keywords"),
            ("interpro", "interpro_annotations"),
            ("pfam", "pfam_annotations"),
        ]

        # df column, m2m attribute name, m2m model attribute to use as
        # string value
        for interaction in query:
            source = interaction.source.identifier.identifier
            target = interaction.target.identifier.identifier
            index.append(",".join(sorted([source, target])))

            data = {"source": source, "target": target}

            for (column, attr) in features:
                joined = ",".join(
                    sorted(
                        [
                            str(a)
                            for a in getattr(interaction.source.record, attr)
                        ]
                        + [
                            str(a)
                            for a in getattr(interaction.target.record, attr)
                        ]
                    )
                )
                data[column] = joined or None

            # Other interaction metadata not used during training
            data["database"] = (
                ",".join(sorted([str(db) for db in interaction.databases]))
                or None
            )
            data["label"] = (
                ",".join(sorted([str(l) for l in interaction.labels])) or None
            )
            data["evidence"] = (
                ",".join(sorted([str(e) for e in interaction.evidence]))
                or None
            )

            interactions.append(data)

        return pd.DataFrame(
            data=interactions,
            columns=[
                "source",
                "target",
                "go_mf",
                "go_bp",
                "go_cc",
                "keyword",
                "interpro",
                "pfam",
                "label",
                "evidence",
                "database",
            ],
            index=index,
        )


class ClassifierModel(BaseModel):
    name = peewee.TextField(
        null=False,
        default=None,
        unique=True,
        help_text="The name of the model.",
    )
    model = playhouse.fields.PickleField(
        null=False, default=None, help_text="Pickled Scikit model."
    )


class InteractionPrediction(BaseModel):
    interaction = peewee.ForeignKeyField(
        model=Interaction,
        null=False,
        default=None,
        backref="predictions",
        on_delete=ForeignKeyConstraint.CASCADE,
        help_text="Interaction prediction is for.",
    )
    label = peewee.ForeignKeyField(
        model=InteractionLabel,
        null=True,
        default=None,
        backref="predictions",
        on_delete=ForeignKeyConstraint.CASCADE,
        help_text="Predicted label.",
    )
    probability = peewee.FloatField(
        null=False,
        default=None,
        help_text="Predction probability for this label.",
    )
    model = peewee.ForeignKeyField(
        model=ClassifierModel,
        null=False,
        default=None,
        backref="interactions",
        on_delete=ForeignKeyConstraint.CASCADE,
        help_text="The model used to make this prediction.",
    )
