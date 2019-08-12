from typing import List, Optional

import pandas as pd
import peewee
import playhouse.fields

from ...settings import DATABASE
from .base import BaseModel
from .identifiers import PsimiIdentifier, PubmedIdentifier
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

    def save(self, *args, **kwargs):
        self.name = self.name.strip().capitalize()
        return super().save(*args, **kwargs)


class InteractionLabel(BaseModel):
    text = peewee.CharField(
        null=False,
        default=None,
        unique=True,
        max_length=64,
        help_text="Interaction type labels.",
    )

    def save(self, *args, **kwargs):
        self.text = self.text.strip().capitalize()
        return super().save(*args, **kwargs)


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
        metadata = [
            ("database", "databases", "name"),
            ("label", "labels", "text"),
        ]
        for interaction in queryset:
            source = interaction.source.identifier.identifier
            target = interaction.target.identifier.identifier
            index.append(",".join(sorted([source, target])))

            data = {"source": source, "target": target}

            for (column, attr) in features:
                joined = ",".join(
                    sorted(
                        [a.to_str() for a in getattr(interaction.source, attr)]
                        + [
                            a.to_str()
                            for a in getattr(interaction.target, attr)
                        ]
                    )
                )
                data[column] = joined or None

            for (column, attr, rel_attr) in metadata:
                joined = ",".join(
                    sorted(
                        [
                            getattr(elem, rel_attr)
                            for elem in getattr(interaction, attr)
                        ]
                    )
                )
                data[column] = joined or None

            # Format evidence
            data["evidence"] = (
                ",".join(
                    sorted(
                        "{}|{}".format(
                            evidence.pubmed.identifier,
                            getattr(evidence.psimi, "identifier", None),
                        )
                        for evidence in interaction.evidence
                    )
                )
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
