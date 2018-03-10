from ..base.utilities import remove_duplicates

from .exceptions import ObjectAlreadyExists, ObjectNotFound
from .exceptions import NonMatchingTaxonomyIds

__all__ = [
    'format_annotation',
    'format_annotations',
    'format_label',
    'format_labels',
    'validate_annotations',
    'validate_go_annotations',
    'validate_pfam_annotations',
    'validate_interpro_annotations',
    'validate_keywords',
    'validate_function',
    'validate_protein',
    'validate_source_and_target',
    'validate_same_taxonid',
    'validate_labels',
    'validate_boolean',
    'validate_joint_id',
    'validate_interaction_does_not_exist',
    'validate_training_holdout_is_labelled',
    'validate_gene_id',
    'validate_taxon_id',
    'validate_accession',
    'validate_description',
    'validate_accession_does_not_exist',
    'validate_uniprot_does_not_exist'
]


def format_annotation(value, upper=True):
    if not isinstance(value, (str, type(None))):
        raise TypeError("Annotation must be a string or None.")
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if upper:
        return value.upper()
    else:
        return value


def format_annotations(values, upper=True, allow_duplicates=True):
    if not isinstance(values, (str, list, set, type(None))):
        raise TypeError(
            "Annotations must be list, str, set or None. Found {}.".format(
                type(values).__name__)
        )

    if values is None:
        return None
    elif isinstance(values, str):
        values = [v.strip() for v in values.split(',') if v.strip()]
    else:
        values = [v.strip() for v in values if (v is not None) and v.strip()]

    values = [
        format_annotation(value, upper) for value in values
        if format_annotation(value, upper)
    ]
    if not allow_duplicates:
        values = list(sorted(set(values)))
    if not values:
        return None
    return list(sorted(values))


def format_label(value, capitalize=True):
    if not isinstance(value, (str, int, type(None))):
        raise TypeError("Label must be a string, int or None.")
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if capitalize:
        return value.capitalize()
    else:
        return value


def format_labels(values, capitalize=True):
    if not isinstance(values, (str, list, set, type(None))):
        raise TypeError(
            "Label must be list, str, set or None. Found {}.".format(
                type(values).__name__)
        )
    if values is None:
        return None
    elif isinstance(values, str):
        values = [v.strip() for v in values.split(',') if v.strip()]

    values = list(sorted(set([
        format_label(value, capitalize) for value in values
        if format_label(value, capitalize)
    ])))
    if not values:
        return None
    return values


def validate_annotations(values, dbtype, upper=True, allow_duplicates=True):
    values = format_annotations(values, upper, allow_duplicates)
    if not values:
        return None

    if dbtype == "GO":
        all_valid = all(["GO:" in v for v in values])
    elif dbtype == "INTERPRO":
        all_valid = all(["IPR" in v for v in values])
    elif dbtype == "PFAM":
        all_valid = all(["PF" in v for v in values])
    else:
        raise ValueError("Unrecognised dbtype '{}'".format(dbtype))

    if not all_valid:
        raise ValueError(
            "Annotations contain invalid values for database type {}. "
            "Caused by {}.".format(
                dbtype, values
            )
        )
    return values


def validate_go_annotations(values, upper=True, allow_duplicates=False):
    values = validate_annotations(
        values, dbtype='GO',
        upper=upper, allow_duplicates=allow_duplicates
    )
    if not values:
        return None
    return ','.join(values)


def validate_pfam_annotations(values, upper=True, allow_duplicates=False):
    values = validate_annotations(
        values, dbtype='PFAM',
        upper=upper, allow_duplicates=allow_duplicates
    )
    if not values:
        return None
    return ','.join(values)


def validate_interpro_annotations(values, upper=True, allow_duplicates=False):
    values = validate_annotations(
        values, dbtype='INTERPRO',
        upper=upper, allow_duplicates=allow_duplicates
    )
    if not values:
        return None
    return ','.join(values)


def validate_keywords(values, allow_duplicates=False):
    if not isinstance(values, (str, list, set, type(None))):
        raise TypeError(
            "Keywords must be list, str, set or None. Found {}.".format(
                type(values).__name__)
        )
    if values is None:
        return None
    elif isinstance(values, str):
        values = values.split(',')
    else:
        values = list(values)

    values = [
        v.capitalize().strip() for v in values
        if (v is not None) and v.strip()
    ]
    if not values:
        return None
    else:
        if allow_duplicates:
            return ','.join(list(sorted(values)))
        else:
            return ','.join(list(sorted(set(values))))


def validate_function(value):
    if not isinstance(value, (str, type(None))):
        raise TypeError(
            "Function must be str or None. Found {}.".format(
                type(value).__name__)
        )
    if value is None:
        return None
    elif not value.strip():
        return None
    else:
        return value.strip()


def validate_description(value):
    if not isinstance(value, (str, type(None))):
        raise TypeError(
            "Description must be string or None. Found {}.".format(
                type(value).__name__)
        )
    if value is None:
        return None
    elif not value.strip():
        return None
    else:
        return value.strip()


def validate_protein(value, return_instance=False):
    from .models import Protein
    if isinstance(value, Protein):
        value_id = value.id
        entry = value

    elif isinstance(value, int):
        entry = Protein.query.get(value)
        if entry is None:
            raise ObjectNotFound("Protein {} does not exist.".format(value))
        else:
            value_id = value

    elif isinstance(value, str):
        entry = Protein.query.filter_by(uniprot_id=value).first()
        if entry is None:
            raise ObjectNotFound("Protein {} does not exist.".format(value))
        else:
            value_id = entry.id
    else:
        raise TypeError(
            "`value` must be a Protein, uniprot string or int id."
        )

    if value_id is None:
        raise ValueError("Protein must be first saved to generate an id.")

    if return_instance:
        return entry
    return value_id


def validate_source_and_target(source, target):
    source = validate_protein(source)
    target = validate_protein(target)
    return source, target


def validate_same_taxonid(source, target):
    source = validate_protein(source, return_instance=True)
    target = validate_protein(target, return_instance=True)
    id_s = source.taxon_id
    id_t = target.taxon_id
    if id_s != id_t:
        raise NonMatchingTaxonomyIds(
            "Proteins do not have matching taxonomy ids {}, {}.".format(
                id_s, id_t
            )
        )
    return id_s


def validate_labels(values, return_list=False):
    value = format_labels(values)
    if not value:
        return None if not return_list else []
    if return_list:
        return value
    return ','.join(format_labels(values))


def validate_boolean(value):
    if value is None:
        return False
    if not isinstance(value, bool):
        raise TypeError("value `{}` must be a boolean.".format(value))
    return value


def validate_joint_id(source, target):
    if not isinstance(source, int):
        raise TypeError("Source must be an int id.")
    if not isinstance(target, int):
        raise TypeError("Target must be an int id.")
    return ','.join([str(x) for x in sorted([source, target])])


def validate_interaction_does_not_exist(source, target):
    from .models import Interaction
    source = validate_protein(source, return_instance=True)
    target = validate_protein(target, return_instance=True)
    joint_id = validate_joint_id(source.id, target.id)
    if Interaction.query.filter(Interaction.joint_id_ == joint_id).first():
        raise ObjectAlreadyExists(
            "Interaction ({}, {}) already exists.".format(
                source.uniprot_id, target.uniprot_id)
        )


def validate_training_holdout_is_labelled(label, is_holdout, is_training):
    if label is None and (is_holdout or is_training):
        raise ValueError("Holdout/Training interaction must be labelled")


def validate_gene_id(gene_id):
    if not isinstance(gene_id, (str, type(None))):
        raise TypeError("`gene_id` must be a str or None")
    if gene_id is None:
        return None
    elif not gene_id.strip():
        return None
    else:
        return gene_id.strip()


def validate_taxon_id(id_):
    if not isinstance(id_, int):
        raise TypeError("`taxon_id` must be an int")
    return id_


def validate_uniprot_does_not_exist(value, klass):
    if not hasattr(klass, 'uniprot_id'):
        raise AttributeError(
            "Class {} does not have attr `uniprot_id`.".format(klass.__name__)
        )
    if klass.get_by_uniprot_id(value) is not None:
        raise ObjectAlreadyExists(
            "A Protein entry with the uniprot id '{}' already exists.".format(
                value
            )
        )


def validate_accession_does_not_exist(value, klass):
    if not hasattr(klass, 'accession'):
        raise AttributeError(
            "Class {} does not have attr `accession`.".format(klass.__name__)
        )
    if klass.query.filter(klass.accession == value).first():
        raise ObjectAlreadyExists(
            "A {} entry with the accession '{}' already exists.".format(
                klass.__name__, value
            )
        )


def validate_accession(value, klass=None, upper=True, check_exists=True):
    if not isinstance(value, str):
        raise TypeError("Accession must be a string.")
    elif not value.strip():
        raise ValueError("Accession must be a non-empty string.")

    if upper:
        value = value.strip().upper()
    else:
        value = value.strip()

    if check_exists:
        if klass.__name__ == 'Protein':
            validate_uniprot_does_not_exist(value, klass)
        else:
            validate_accession_does_not_exist(value, klass)
    return value
