import datetime
import json
from typing import Generator, Iterable, List, Optional, Tuple, Union

import pandas as pd
import peewee
import playhouse.fields

from ..constants import GeneOntologyCategory
from ..settings import DATABASE


class BaseModel(peewee.Model):
    FORMAT = "%Y-%m-%d %H:%M:%S.%f"

    created = peewee.DateTimeField()
    modified = peewee.DateTimeField()

    class Meta:
        database = DATABASE

    def refresh(self):
        """Refresh from database"""
        return self.__class__.get(self._pk_expr())

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
    # Database name of the identifier.
    DB_NAME: Optional[str] = None
    # Prefix for appending to an identifier if missing. For example the
    # GO in GO:<accession>.
    PREFIX: Optional[str] = None
    # How to separate identifier and prefix. For example the ':' between
    # GO and <accession>.
    SEP: Optional[str] = ":"

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
    SEP = ":"


class PubmedIdentifier(ExternalIdentifier):
    DB_NAME = "PubMed"
    PREFIX = "PMID"
    SEP = ":"


class PsimiIdentifier(ExternalIdentifier):
    DB_NAME = "Psimi"
    PREFIX = "MI"
    SEP = ":"


class UniprotIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt"
    PREFIX = None
    SEP = None


class KeywordIdentifier(ExternalIdentifier):
    DB_NAME = "UniProt KW"
    PREFIX = "KW"
    SEP = "-"


class InterproIdentifier(ExternalIdentifier):
    DB_NAME = "InterPro"
    PREFIX = None
    SEP = None


class PfamIdentifier(ExternalIdentifier):
    DB_NAME = "PFAM"
    PREFIX = None
    SEP = None


class AnnotationMixin:
    def to_str(self) -> str:
        return getattr(self, "identifier").identifier


class GeneOntologyTerm(BaseModel, AnnotationMixin):
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
        null=True, default=None, help_text="GO term description."
    )
    category = peewee.CharField(
        null=False,
        default=None,
        help_text="The GO category that the term belongs to.",
    )
    obsolete = peewee.BooleanField(
        null=True,
        default=None,
        help_text="Term is obsolete according to the GO.",
    )

    def save(self, *args, **kwargs):
        self.category = self.category.strip().capitalize()
        if len(self.category) == 1:
            self.category = GeneOntologyCategory.letter_to_category(
                self.category
            )
        if self.category not in set(GeneOntologyCategory.list()):
            raise ValueError(
                f"'{self.category}' is not a supported category. "
                f"Supported categories are {GeneOntologyCategory.list()}"
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
    entry_type = peewee.CharField(
        null=True,
        default=None,
        help_text="Interpro entry type.",
        max_length=32,
    )

    @classmethod
    @DATABASE.atomic()
    def update_or_create(
        cls,
        identifier: Union[str, GeneOntologyIdentifier],
        name: str,
        entry_type: Optional[str] = None,
    ):
        if isinstance(identifier, str):
            identifier, _ = InterproIdentifier.get_or_create(
                identifier=identifier
            )

        instance, created = cls.get_or_create(
            identifier=identifier,
            defaults={"name": name, "entry_type": entry_type},
        )
        if not created:
            instance.name = name
            if entry_type:
                instance.entry_type = entry_type
            instance.save()

        return instance


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

    @classmethod
    @DATABASE.atomic()
    def update_or_create(
        cls,
        identifier: Union[str, GeneOntologyIdentifier],
        name: str,
        description: Optional[str] = None,
    ):
        if isinstance(identifier, str):
            identifier, _ = PfamIdentifier.get_or_create(identifier=identifier)

        instance, created = cls.get_or_create(
            identifier=identifier,
            defaults={"name": name, "description": description},
        )
        if not created:
            instance.name = name
            if description:
                instance.description = description
            instance.save()

        return instance


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

    @classmethod
    @DATABASE.atomic()
    def update_or_create(
        cls,
        identifier: Union[str, GeneOntologyIdentifier],
        description: Optional[str] = None,
    ):
        if isinstance(identifier, str):
            identifier, _ = PfamIdentifier.get_or_create(identifier=identifier)

        instance, created = cls.get_or_create(
            identifier=identifier, defaults={"description": description}
        )
        if not created:
            if description:
                instance.description = description
            instance.save()

        return instance


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
    organism = peewee.IntegerField(
        null=False, default=None, help_text="Numeric organism code. Eg 9606."
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
    gene = peewee.ForeignKeyField(
        model=GeneSymbol,
        null=True,
        default=None,
        backref="proteins",
        help_text="Gene symbol related to this protein.",
    )
    alt_genes = peewee.ManyToManyField(
        model=GeneSymbol, backref="proteins_alt"
    )
    sequence = peewee.TextField(
        null=False,
        default=None,
        help_text="The protein sequence in single letter format.",
    )
    reviewed = peewee.BooleanField(
        null=True,
        default=None,
        help_text="Protein has been reviewed (Swiss-prot).",
    )

    def _select_go(self, category: str):
        return (
            GeneOntologyTerm.select()
            .join(Protein.go_annotations.get_through_model())
            .join(Protein)
            .where(Protein.id == self.id)
            .select()
            .where(GeneOntologyTerm.category == category)
        )

    @property
    def go_mf(self):
        return self._select_go(GeneOntologyCategory.molecular_function)

    @property
    def go_bp(self):
        return self._select_go(GeneOntologyCategory.biological_process)

    @property
    def go_cc(self):
        return self._select_go(GeneOntologyCategory.cellular_compartment)

    @classmethod
    @DATABASE.atomic()
    def update_or_create(
        cls,
        identifier: Union[str, UniprotIdentifier],
        sequence: str,
        organism: int,
        aliases: Iterable[Union[str, UniprotIdentifier]] = (),
        go: Iterable[GeneOntologyTerm] = (),
        interpro: Iterable[InterproTerm] = (),
        pfam: Iterable[PfamTerm] = (),
        keywords: Iterable[Keyword] = (),
        gene: Optional[Union[str, GeneSymbol]] = None,
        alt_genes: Iterable[Union[str, GeneSymbol]] = (),
    ):
        if isinstance(identifier, str):
            identifier = UniprotIdentifier.get_or_create(
                identifier=identifier
            )[0]

        query = Protein.get_or_create(identifier=identifier)
        instance: Protein = query[0]
        instance.sequence = sequence
        instance.aliases.add(
            [
                alias
                if isinstance(alias, UniprotIdentifier)
                else UniprotIdentifier.get_or_create(identifier=alias)[0]
                for alias in aliases
            ]
        )
        instance.organism = organism

        # Update annotations
        instance.go_annotations.add([term for term in go])
        instance.interpro_annotations.add([term for term in interpro])
        instance.pfam_annotations.add([term for term in pfam])
        instance.keywords.add([term for term in keywords])

        # Update gene symbols
        if isinstance(gene, str):
            instance.gene = GeneSymbol.get_or_create(text=gene)[0]
        else:
            instance.gene = gene
        instance.alt_genes.add(
            [
                gene
                if isinstance(gene, GeneSymbol)
                else GeneSymbol.get_or_create(text=gene)[0]
                for gene in alt_genes
            ]
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

    class Meta:
        indexes = (
            # create a unique on source/target
            (("source_id", "target_id"), True),
        )

    @classmethod
    @DATABASE.atomic()
    def update_or_create(
        cls,
        source: Protein,
        target: Protein,
        organism: int = None,
        labels: Iterable[Union[str, InteractionLabel]] = (),
        psimi_ids: Iterable[Union[str, PsimiIdentifier]] = (),
        pubmed_ids: Iterable[Union[str, PubmedIdentifier]] = (),
        experiment_types: Iterable[Union[str, ExperimentType]] = (),
        databases: Iterable[Union[str, InteractionDatabase]] = (),
    ):
        """
        Searches for an existing interaction given the source, target and
        organism. Treats source and target as interchangeable. Adds
        additional metadata in `kwargs` to existing interactions, such as 
        new labels, pmids etc. 
        
        Metadata supplied as strings will be created if they don't already 
        exist.
        
        Args:
            source (Protein): 
                Source protein instance.
            
            target (Protein): 
                Target protein instance.
            
            organism (int, optional): 
                Numeric organism code. Defaults to 9606.
            
            labels (Iterable[Union[str, InteractionLabel]], optional): 
                Labels as strings or instances. Will be appened to existing 
                instances. Defaults to ().
            
            psimi_ids (Iterable[Union[str, PsimiIdentifier]], optional): 
                PSI-MI identifiers as strings or instances. Will be appened to 
                existing instances. Defaults to ().
            
            pubmed_ids (Iterable[Union[str, PubmedIdentifier]], optional): 
                PubMed IDs as strings or instances. Will be appened to existing 
                instances. Defaults to ().
            
            experiment_types (Iterable[Union[str, ExperimentType]], optional): 
                Experiment types as strings or instances. Will be appened to 
                existing instances. Defaults to ().
            
            databases (Iterable[Union[str, InteractionDatabase]], optional): 
                Database as strings or instances. Will be appened to existing 
                instances. Defaults to ().

        Returns:
            Interaction: Newly created interaction with supplied parameters,
            or an existing interaction with updates fields.
        """
        result = cls.get_or_create(source=source, target=target)
        instance: Interaction = result[0]

        # Change organism if provided
        if organism is not None:
            instance.organism = organism

        # Add additional labels.
        instance.labels.add(
            [
                label
                if isinstance(label, InteractionLabel)
                else InteractionLabel.get_or_create(text=label)[0]
                for label in labels
            ]
        )
        # Add pubmed PSI-MI assay description identifiers.
        instance.psimi_ids.add(
            [
                ident
                if isinstance(ident, PsimiIdentifier)
                else PsimiIdentifier.get_or_create(identifier=ident)[0]
                for ident in psimi_ids
            ]
        )
        # Add pubmed evidence identifiers.
        instance.pubmed_ids.add(
            [
                ident
                if isinstance(ident, PubmedIdentifier)
                else PubmedIdentifier.get_or_create(identifier=ident)[0]
                for ident in pubmed_ids
            ]
        )
        # Add pubmed experiment type descriptors.
        instance.experiment_types.add(
            [
                etype
                if isinstance(etype, ExperimentType)
                else ExperimentType.get_or_create(text=etype)[0]
                for etype in experiment_types
            ]
        )
        # Add databases in which interaction appears.
        instance.databases.add(
            [
                name
                if isinstance(name, InteractionDatabase)
                else InteractionDatabase.get_or_create(name=name)[0]
                for name in databases
            ]
        )
        instance.save()
        return instance

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
            ("psimi", "psimi_ids", "identifier"),
            ("pmid", "pubmed_ids", "identifier"),
            ("experiment_type", "experiment_types", "text"),
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
                "psimi",
                "pmid",
                "experiment_type",
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
    Protein.alt_genes.get_through_model(),
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
