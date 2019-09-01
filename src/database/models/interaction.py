from typing import List, Optional, Tuple, Iterable
from functools import reduce

import pandas as pd
import peewee
import playhouse.fields

from ...settings import DATABASE
from .base import BaseModel
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
        help_text="Pubmed identifier supporting an interaction.",
    )
    psimi = peewee.ForeignKeyField(
        model=PsimiIdentifier,
        null=True,
        default=None,
        backref="evidence_set",
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
    def filter_by_index(
        cls,
        identifiers: List[Tuple[str, Optional[str]]],
        query: Optional[peewee.ModelSelect] = None,
    ) -> peewee.ModelSelect:
        """
        Filter evidence table for tuples with matching pubmed and psimi 
        identifiers. Identifiers will be formatted prior to filtering.
        
        Parameters
        ----------
        identifiers : List[Tuple[str, str]]
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
        help_text="Source protein.",
    )
    target = peewee.ForeignKeyField(
        model=Protein,
        null=False,
        default=None,
        backref="interactions_as_target",
        help_text="Target protein.",
    )

    # Interaction metadata.
    organism = peewee.IntegerField(
        null=True,
        default=None,
        help_text=(
            "Numeric organism code. Eg 9606. This can be None if the "
            "interaction is a species hybrid."
        ),
    )
    labels = peewee.ManyToManyField(
        model=InteractionLabel, backref="interactions"
    )

    # Fields relating to evidence/experiment detection method.
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
    def filter_by_index(
        cls,
        identifiers: List[Tuple[str, str]],
        query: Optional[peewee.ModelSelect] = None,
    ) -> peewee.ModelSelect:
        cls.bulk_create
        """
        Filter by source and target identifiers. Identifiers will be formatted 
        prior to filtering.
        
        Parameters
        ----------
        identifiers : List[Tuple[str, str]]
            Uniprot identifiers using a (source, target) format.
        query : Optional[ModelSelect], optional
            Query to filter. Defaults to all rows.
        
        Returns
        -------
        peewee.ModelSelect
        """
        query = query or cls.all()
        # Create table alias for target column (can't use `UniprotIdentifier`
        # column twice)
        UA = UniprotIdentifier.alias()
        # Create queries to search for each tuple using an AND query.
        queries = [
            (
                (
                    peewee.fn.Upper(UniprotIdentifier.identifier)
                    == UniprotIdentifier.format(source)
                )
                & (
                    peewee.fn.Upper(UA.identifier)
                    == UniprotIdentifier.format(target)
                )
            )
            for source, target in identifiers
        ]
        # Search all tuple element using an OR query.
        return (
            query.select(cls, UniprotIdentifier, UA)
            .join(UniprotIdentifier, on=(cls.source == UniprotIdentifier.id))
            .switch(cls)
            .join(UA, on=(cls.target == UA.id).alias("target"))
            .where(reduce(lambda x, y: x | y, queries[1:], queries[0]))
            .switch(cls)
            .select(cls)  # Context switch back to interaction.
        )

    @classmethod
    def to_dataframe(
        cls, queryset: Optional[peewee.ModelSelect] = None
    ) -> pd.DataFrame:
        if queryset is None:
            queryset = cls.select()

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
        for interaction in queryset:
            source = interaction.source.identifier.identifier
            target = interaction.target.identifier.identifier
            index.append(",".join(sorted([source, target])))

            data = {"source": source, "target": target}

            for (column, attr) in features:
                joined = ",".join(
                    sorted(
                        [str(a) for a in getattr(interaction.source, attr)]
                        + [str(a) for a in getattr(interaction.target, attr)]
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
        help_text="Interaction prediction is for.",
    )
    label = peewee.ForeignKeyField(
        model=InteractionLabel,
        null=True,
        default=None,
        backref="predictions",
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
        help_text="The model used to make this prediction.",
    )
