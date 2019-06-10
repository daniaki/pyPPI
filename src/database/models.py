import json
import datetime

from typing import Optional, Tuple, List, Generator

import peewee
import playhouse.fields

from ..settings import DATABASE


class BaseModel(peewee.Model):
    FORMAT = "%Y-%m-%d %H:%M:%S.%f"

    created = peewee.DateTimeField()
    modified = peewee.DateTimeField()

    class Meta:
        database = DATABASE

    def save(self, *args, **kwargs):
        if self.created is None:
            self.created = datetime.datetime.now()
        self.modified = datetime.datetime.now()
        return super().save(*args, **kwargs)


class IdentifierMixin:
    def _get_identifier(self) -> str:
        identifier: Optional[str] = getattr(self, "identifier", None)
        if not getattr(self, "identifier", None):
            raise AttributeError(
                f"{self.__class__.__name__} is missing attribute 'identifier'."
            )
        if not isinstance(identifier, str):
            klass = type(identifier).__name__
            raise TypeError(
                f"Expected 'identifier' to be 'str'. Found '{klass}'"
            )
        return identifier

    def prefix(self, prefix: str, sep: str = ":") -> str:
        identifier = self._get_identifier()
        if not identifier.lower().startswith(f"{prefix.lower()}{sep}"):
            return f"{prefix}{sep}{identifier}"
        return identifier

    def unprefix(self, sep: str = ":") -> str:
        return self._get_identifier().split(sep)[-1]


class ExternalIdentifier(IdentifierMixin, BaseModel):
    DB_NAME: Optional[str] = None
    PREFIX: Optional[str] = None
    SEP = ":"

    identifier = peewee.CharField(
        null=False,
        default=None,
        unique=True,
        max_length=32,
        help_text="The unique identifier from an external database.",
    )
    dbname = peewee.CharField(
        null=False,
        default=None,
        max_length=16,
        help_text="The identifier's database name.",
    )

    def save(self, *args, **kwargs):
        if self.DB_NAME is None:
            raise NotImplementedError("Concrete table must define DB_NAME.")
        if self.PREFIX:
            self.identifier = self.prefix(self.PREFIX, self.SEP)
        self.identifier = self.identifier.upper()
        self.dbname = self.DB_NAME
        return super().save(*args, **kwargs)


class GeneOntologyIdentifier(ExternalIdentifier):
    DB_NAME = "Gene Ontology"
    PREFIX = "GO"


class PubmedIdentifier(ExternalIdentifier):
    DB_NAME = "PubMed"
    PREFIX = "PMID"


class PsimiIdentifier(ExternalIdentifier):
    DB_NAME = "Psimi"
    PREFIX = "MI"


class UniprotIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt"
    PREFIX = None


class KeywordIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt KW"
    PREFIX = "KW"
    SEP = "-"


class InterproIdentifier(ExternalIdentifier):
    DB_NAME = "InterPro"
    PREFIX = None


class PfamIdentifier(ExternalIdentifier):
    DB_NAME = "PFAM"
    PREFIX = None


class AnnotationMixin:
    def to_str(self) -> str:
        return getattr(self, "identifier").identifier


class GeneOntologyTerm(BaseModel, AnnotationMixin):
    class Category:
        molecular_function = "Molecular function"
        biological_process = "Biological process"
        cellular_compartment = "Cellular compartment"

        @classmethod
        def list(cls):
            return [
                cls.molecular_function,
                cls.biological_process,
                cls.cellular_compartment,
            ]

        @classmethod
        def letter_to_category(cls, letter: str) -> str:
            if letter.upper() == "C":
                return cls.cellular_compartment
            elif letter.upper() == "P":
                return cls.biological_process
            elif letter.upper() == "F":
                return cls.molecular_function
            else:
                raise ValueError(
                    f"'{letter}' is not a supported shorthand category."
                )

        @classmethod
        def choices(cls):
            return [(c, c) for c in cls.list()]

    identifier = peewee.ForeignKeyField(
        model=GeneOntologyIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Unique identifier for this term.",
    )
    name = peewee.TextField(
        null=False, default=None, help_text="GO term name."
    )
    description = peewee.TextField(
        null=False, default=None, help_text="GO term description."
    )
    category = peewee.CharField(
        null=False,
        default=None,
        help_text="The GO category that the term belongs to.",
    )
    obsolete = peewee.BooleanField(
        null=False,
        default=False,
        help_text="Term is obsolete according to the GO.",
    )

    def save(self, *args, **kwargs):
        if len(self.category) == 1:
            self.category = self.Category.letter_to_category(self.category)
        self.category = self.category.strip().capitalize()
        if self.category not in set(self.Category.list()):
            raise ValueError(
                f"'{self.category}' is not a supported category. "
                f"Supported categories are {self.Category.list()}"
            )
        return super().save(*args, **kwargs)


class InterproTerm(BaseModel, AnnotationMixin):
    identifier = peewee.ForeignKeyField(
        model=InterproIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Identifier relating to this term.",
    )
    name = peewee.TextField(
        null=False, default=None, help_text="Short name description of a term."
    )
    description = peewee.TextField(
        null=False, default=None, help_text="Long form description of a term."
    )
    entry_type = peewee.CharField(
        null=True,
        default=None,
        help_text="Interpro entry type.",
        max_length=32,
    )


class PfamTerm(BaseModel, AnnotationMixin):
    identifier = peewee.ForeignKeyField(
        model=PfamIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Identifier relating to this term.",
    )
    name = peewee.TextField(
        null=False, default=None, help_text="Short name description of a term."
    )
    description = peewee.TextField(
        null=False, default=None, help_text="Long form description of a term."
    )


class Keyword(BaseModel, AnnotationMixin):
    identifier = peewee.ForeignKeyField(
        model=KeywordIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="terms",
        help_text="Identifier relating to this term. Has form 'KW-d+'",
    )
    description = peewee.TextField(
        null=False, default=None, help_text="Keyword description."
    )


class GeneSymbol(BaseModel):
    text = peewee.CharField(
        null=False, default=None, unique=True, help_text="HGNC gene symbol."
    )

    def save(self, *args, **kwargs):
        self.text = self.text.strip().upper()
        return super().save(*args, **kwargs)


class ExperimentType(BaseModel):
    text = peewee.CharField(
        null=False,
        default=None,
        unique=True,
        help_text="Experiment/Assay detection method.",
    )

    def save(self, *args, **kwargs):
        self.text = self.text.strip().capitalize()
        return super().save(*args, **kwargs)


class Protein(BaseModel):
    identifier = peewee.ForeignKeyField(
        model=UniprotIdentifier,
        null=False,
        default=None,
        unique=True,
        backref="proteins",
    )
    aliases = peewee.ManyToManyField(
        model=UniprotIdentifier, backref="proteins_aliased"
    )
    go_annotations = peewee.ManyToManyField(
        model=GeneOntologyTerm, backref="proteins"
    )
    interpro_annotations = peewee.ManyToManyField(
        model=InterproTerm, backref="proteins"
    )
    pfam_annotations = peewee.ManyToManyField(
        model=PfamTerm, backref="proteins"
    )
    keywords = peewee.ManyToManyField(model=Keyword, backref="proteins")
    gene_name = peewee.ForeignKeyField(
        model=GeneSymbol,
        null=True,
        default=None,
        backref="proteins",
        help_text="Gene symbol related to this protein.",
    )
    alt_gene_names = peewee.ManyToManyField(
        model=GeneSymbol, backref="proteins_alt"
    )
    sequence = peewee.TextField(
        null=False,
        default=None,
        help_text="The protein sequence in single base format.",
    )

    def save(self, *args, **kwargs):
        self.sequence = self.sequence.strip().upper()
        return super().save(*args, **kwargs)


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
        null=False, default=None, help_text="Numeric organism code. Eg 9606."
    )
    direction = peewee.IntegerField(
        null=False,
        default=1,
        help_text=(
            "Direction of the interaction. Positive for source to target. "
            "Negative for target to source.",
        ),
    )
    labels = peewee.ManyToManyField(
        model=InteractionLabel, backref="interactions"
    )

    # Fields relating to evidence/experiment detection method.
    psimi_ids = peewee.ManyToManyField(
        model=PsimiIdentifier, backref="interactions"
    )
    pubmed_ids = peewee.ManyToManyField(
        model=PubmedIdentifier, backref="interactions"
    )
    experiment_types = peewee.ManyToManyField(
        model=ExperimentType, backref="interactions"
    )
    databases = peewee.ManyToManyField(
        model=InteractionDatabase, backref="interactions"
    )

    # Unique hash for easy identification.
    obj_hash = peewee.CharField(
        max_length=64,
        null=False,
        default=None,
        unique=True,
        help_text=(
            "Hash based on the source, target and organism. "
            "Ensures interaction uniquness."
        ),
    )

    @classmethod
    def get_update_or_create(cls, **kwargs):
        source = kwargs.get("source", None)
        target = kwargs.get("target", None)
        organism = kwargs.get("organism", None)

        cls.get_or_create(source=source, target=target, organism=organism)

    @classmethod
    def format_xy(
        cls,
        queryset: Optional[peewee.ModelSelect] = None,
        features: Tuple[str, str, str] = (
            Protein.go_annotations.name,
            Protein.pfam_annotations.name,
            Protein.interpro_annotations.name,
        ),
    ) -> Generator[Tuple[List[str], List[str]], None, None]:
        if queryset is None:
            queryset = cls.select()

        Interaction
        for interaction in queryset:
            annotations: List[str] = []
            for feature in features:
                # Collect source annotations
                annotations.extend(
                    a.to_str() for a in getattr(interaction.source, feature)
                )
                # Collect target annotations
                annotations.extend(
                    a.to_str() for a in getattr(interaction.target, feature)
                )

            labels = [label.text for label in interaction.labels]
            yield annotations, labels

    def hash(self):
        return hash(
            (tuple(sorted([self.source.id, self.target.id])), self.organism)
        )

    def save(self, *args, **kwargs):
        self.obj_hash = self.hash()
        return super().save(*args, **kwargs)


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


MODELS = (
    # Identifiers
    GeneOntologyIdentifier,
    PfamIdentifier,
    InterproIdentifier,
    UniprotIdentifier,
    PubmedIdentifier,
    PsimiIdentifier,
    KeywordIdentifier,
    # Annotations
    GeneOntologyTerm,
    PfamTerm,
    InterproTerm,
    Keyword,
    GeneSymbol,
    ExperimentType,
    # Protein
    Protein,
    Protein.go_annotations.get_through_model(),
    Protein.aliases.get_through_model(),
    Protein.alt_gene_names.get_through_model(),
    Protein.interpro_annotations.get_through_model(),
    Protein.pfam_annotations.get_through_model(),
    Protein.keywords.get_through_model(),
    # Interaction
    Interaction,
    InteractionDatabase,
    InteractionLabel,
    Interaction.labels.get_through_model(),
    Interaction.pubmed_ids.get_through_model(),
    Interaction.psimi_ids.get_through_model(),
    Interaction.experiment_types.get_through_model(),
    Interaction.databases.get_through_model(),
    InteractionPrediction,
    ClassifierModel,
)
