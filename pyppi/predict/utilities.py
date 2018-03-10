import logging
import numpy as np

import bz2
import gzip

from ..base.utilities import rename
from ..database.models import Interaction
from ..database.utilities import (
    full_training_network, training_interactions,
    interactome_interactions, holdout_interactions
)

from ..models.utilities import publication_ensemble, make_gridsearch_clf
from ..models.binary_relevance import MixedBinaryRelevanceClassifier

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

logger = logging.getLogger("pyppi")


DEFAULT_SELECTION = (
    Interaction.columns().GO_MF.value,
    Interaction.columns().GO_CC.value,
    Interaction.columns().GO_BP.value,
    Interaction.columns().INTERPRO.value,
    Interaction.columns().PFAM.value
)

VALID_SELECTION = (
    Interaction.columns().GO_MF.value,
    Interaction.columns().GO_CC.value,
    Interaction.columns().GO_BP.value,
    Interaction.columns().ULCA_GO_MF.value,
    Interaction.columns().ULCA_GO_CC.value,
    Interaction.columns().ULCA_GO_BP.value,
    Interaction.columns().INTERPRO.value,
    Interaction.columns().PFAM.value
)


def paper_model(labels, rcv_splits=3, rcv_iter=30, scoring='f1',
                n_jobs_model=1, n_jobs_br=1, n_jobs_gs=1, random_state=42):
    """This creates a :class:`MixedBinaryRelevanceClassifier`. A `Pipeline`
    classifier with the estimator step being a `RandomizedGridSearch`
    classifier is created. The estimator inside the grid search for
    each label are those specified by :func:`publication_ensemble`.

    Parameters:
    ----------
    labels : list
        List of labels which will be used to initialise a `Binary Relevance`
        classifier with.

    rcv_splits : int, optional, default: 3
        The number of splits to use during hyper-parameter cross-validation.

    rcv_iter : int, optional, default: 30
        The number of grid search iterations to perform.

    scoring : str, optional default: f1
        Scoring method used during hyperparameter search.

    n_jobs_model : int, optional, default: 1
        Sets the `n_jobs` parameter of the Pipeline's estimator step.

    n_jobs_br : {int}, optional
        Sets the `n_jobs` parameter of the :class:`MixedBinaryRelevanceClassifier`
        step.

    n_jobs_gs : int, optional, default: 1
        Sets the `n_jobs` parameter of the `RandomizedGridSearch` classifier.

    random_state : int or None, optional, default: None
        This is a seed used to generate random_states for all estimator
        objects such as the base model and the grid search.

    Returns
    -------
    :class:`MixedBinaryRelevanceClassifier`
        A fully initialised classifier.
    """
    model_dict = publication_ensemble()
    estimators = []
    for label in labels:
        if str(label).capitalize() not in model_dict:
            logger.warning(
                "New label '{}' encountered. Defaulting to "
                "LogisticRegression.".format(label)
            )
        model = model_dict.get(label, 'LogisticRegression')
        rcv = make_gridsearch_clf(
            model, rcv_splits=rcv_splits, rcv_iter=rcv_iter, scoring=scoring,
            n_jobs_model=n_jobs_model, n_jobs_gs=n_jobs_gs,
            search_vectorizer=True, random_state=random_state
        )
        estimators.append(rcv)
    return MixedBinaryRelevanceClassifier(estimators, n_jobs=n_jobs_br)


def validation_model(labels, model='LogisticRegression', rcv_splits=3,
                     rcv_iter=30, scoring='f1', binary=True, n_jobs_br=1,
                     n_jobs_model=1, n_jobs_gs=1, random_state=42):
    """This creates a :class:`MixedBinaryRelevanceClassifier`. A `Pipeline`
    classifier with the estimator step being a `RandomizedGridSearch`
    classifier is created. The estimator inside the grid search is that
    specified by `model`

    Parameters:
    ----------
    labels : list
        List of labels which will be used to initialise a `Binary Relevance`
        classifier with.

    model: str
        String class name of the `SciKit-Learn` model which will be the
        `estimator` within the `Pipeline`.

    rcv_splits : int, optional, default: 3
        The number of splits to use during hyper-parameter cross-validation.

    rcv_iter : int, optional, default: 30
        The number of grid search iterations to perform.

    scoring : str, optional default: 'f1'
        Scoring method used during hyperparameter search.

    binary : bool, optional, default: True
        If True sets the `binary` attribute of the `CountVectorizer` to True.

    n_jobs_model : int, optional, default: 1
        Sets the `n_jobs` parameter of the Pipeline's estimator step.

    n_jobs_br : {int}, optional
        Sets the `n_jobs` parameter of the :class:`MixedBinaryRelevanceClassifier`
        step.

    n_jobs_gs : int, optional, default: 1
        Sets the `n_jobs` parameter of the `RandomizedGridSearch` classifier.

    random_state : int or None, optional, default: None
        This is a seed used to generate random_states for all estimator
        objects such as the base model and the grid search.

    Returns
    -------
    :class:`MixedBinaryRelevanceClassifier`
        A fully initialised classifier.
    """

    estimators = []
    for _ in labels:
        rcv = make_gridsearch_clf(
            model, rcv_splits=rcv_splits, rcv_iter=rcv_iter,
            scoring=scoring, n_jobs_gs=n_jobs_gs, binary=binary,
            n_jobs_model=n_jobs_model, search_vectorizer=False,
            random_state=random_state
        )
        estimators.append(rcv)
    return MixedBinaryRelevanceClassifier(estimators, n_jobs=n_jobs_br)


def interactions_to_Xy_format(interactions, selection):
    """Takes a list of :class:`Interaction` instances and converts them
    into `X, y` format. No vectorisation or binarisation is computed
    during this function.

    Parameters:
    ----------
    interactions : list
        List of :class:`Interaction` instances.

    selection : list
        List of annotations to use. Select from 'go_mf', 'go_cc', 'go_bp',
        'ulca_go_mf', 'ulca_go_cc', 'ulca_go_bp', 'interpro', 'pfam'.

    Returns
    -------
    `tuple`
        X and y. X is a list of comma delimited strings containing the features
        from each interaction. Y is a list of string labels taken straight
        from each instance. Both X and y have length n_interactions.
    """

    X = list(range(len(list(interactions))))  # pre-allocate
    y = list(range(len(list(interactions))))
    for i, interaction in enumerate(interactions):
        x = ''
        label = interaction.label
        for attr in selection:
            try:
                value = getattr(interaction, attr)
            except:
                value = getattr(interaction, attr.value)
            if value:
                if x:
                    x = ','.join([x, value])
                else:
                    x = value

        if label is None:
            label = []
        else:
            label = label.split(',')

        X[i] = x.replace(":", "")
        y[i] = label

    return np.asarray(X), y


def load_dataset(interactions, labels=None, selection=DEFAULT_SELECTION):
    """Takes a list of :class:`Interaction` instances and converts them
    into `X, y` format. No vectorisation of the textual features is computed
    during this function. If `labels` is supplied, the `y` is returned
    as an `indicator matrix`.

    Parameters:
    ----------
    interactions : list
        List of :class:`Interaction` instances.

    labels : list, optional, default: None
        A list of labels that appear in the interactions. If supplied, the
        labels are sorted and used to initialise a :class:`MultiLabelBinarizer`,
        which is then used to transform the text labels of the interactions
        into binary `indicator-matrix` format.

    selection : list
        List of annotations to use. Select from 'go_mf', 'go_cc', 'go_bp',
        'ulca_go_mf', 'ulca_go_cc', 'ulca_go_bp', 'interpro', 'pfam'.

    Returns
    -------
    `tuple`
        The first element is X. X is a list of comma delimited strings
        containing the text features from each interaction. The second element
        `y` is either a list of string labels taken straight from each instance,
        or array-like of size (n_interactions, n_labels) if `labels` is not None.
        If labels is supplied, the MultiLabelBinarizer is also returned as the
        third element.
    """

    if not interactions:
        return None, None

    if labels:
        X, y = interactions_to_Xy_format(interactions, selection)
        mlb = MultiLabelBinarizer(classes=sorted(labels))
        y = mlb.fit_transform(y)
        return X, y, mlb
    else:
        X, y = interactions_to_Xy_format(interactions, selection)
        return X, y


def load_training_dataset(taxon_id=9606, selection=DEFAULT_SELECTION):
    """Loads the :func:`full_training_network` and converts these into
    X, the textual features of each interaction as defined by `selection`,
    and y, the binary multi-label indicator matrix output by a
    :class:`MultiLabelBinarizer`. The binarizer and parsed labels
    are also returned.

    Parameters:
    ----------
    taxon_id : int, optional, default: 9606
        Load the training data for a specific organism. Removes all
        training interactions that do not match this value. Ignored if None.

    selection : list
        List of annotations to use. Select from 'go_mf', 'go_cc', 'go_bp',
        'ulca_go_mf', 'ulca_go_cc', 'ulca_go_bp', 'interpro', 'pfam'.

    Returns
    -------
    `dict`
        The data dictionary. The key `training` has the (X, y) tuple, `labels`
        points to the sorted list of labels parsed from the interactions and
        `binarizer` points to the fitted `MultiLabelBinarizer`.
    """
    training = full_training_network(taxon_id)
    labels = set()
    for interaction in training.all():
        labels |= set(interaction.labels_as_list)

    if not training.count():
        return {}

    X_train, y_train = interactions_to_Xy_format(training, selection)
    mlb = MultiLabelBinarizer(classes=sorted(labels))
    y_train = mlb.fit_transform(y_train)
    return {
        "training": (X_train, y_train),
        'labels': list(sorted(labels)),
        'binarizer': mlb
    }


def load_validation_dataset(taxon_id=9606, selection=DEFAULT_SELECTION):
    """Loads all training and holdout interactions and converts these into
    X, the textual features of each interaction as defined by `selection`,
    and y, the binary multi-label indicator matrix output by a
    :class:`MultiLabelBinarizer`. The binarizer and parsed labels
    are also returned. The labels are parsed from the interactions that
    are strictly training only.

    Parameters:
    ----------
    taxon_id : int, optional, default: 9606
        Load the training/holdout data for a specific organism. Removes all
        interactions that do not match this value. Ignored if None.

    selection : list
        List of annotations to use. Select from 'go_mf', 'go_cc', 'go_bp',
        'ulca_go_mf', 'ulca_go_cc', 'ulca_go_bp', 'interpro', 'pfam'.

    Returns
    -------
    `dict`
        The data dictionary. The key `training` has the (X, y) tuple, `labels`
        points to the sorted list of labels parsed from the interactions and
        `binarizer` points to the fitted `MultiLabelBinarizer`. If holdout interactions
        were found, the key `testing` points to the (X, y) tuple for the training
        samples.
    """
    training = training_interactions(strict=True, taxon_id=taxon_id)
    labels = set()
    for interaction in training.all():
        labels |= set(interaction.labels_as_list)

    testing = holdout_interactions(strict=True, taxon_id=taxon_id)
    data = {'labels': list(sorted(labels))}

    if not training.count():
        return {}
    else:
        X_train, y_train = interactions_to_Xy_format(training, selection)
        mlb = MultiLabelBinarizer(classes=sorted(labels))
        y_train = mlb.fit_transform(y_train)
        data['training'] = (X_train, y_train)
        data['binarizer'] = mlb

    if testing.count():
        X_test, y_test = interactions_to_Xy_format(testing, selection)
        y_test = mlb.transform(y_test)
        data['testing'] = (X_test, y_test)

    return data


def load_interactome_dataset(taxon_id=9606, selection=DEFAULT_SELECTION):
    """Loads the :func:`interactome_interactions` and converts these into
    X, the textual features of each interaction as defined by `selection`.

    Parameters:
    ----------
    taxon_id : int, optional, default: 9606
        Load the interactome data for a specific organism. Removes all
        interactions that do not match this value. Ignored if None.

    selection : list
        List of annotations to use. Select from 'go_mf', 'go_cc', 'go_bp',
        'ulca_go_mf', 'ulca_go_cc', 'ulca_go_bp', 'interpro', 'pfam'.

    Returns
    -------
    `list`
        List of str textual features for each interaction.
    """
    interactome = interactome_interactions(taxon_id)
    X_test, _ = interactions_to_Xy_format(interactome, selection)
    return X_test


def train_paper_model(rcv_splits=3, rcv_iter=30, scoring='f1', n_jobs_model=1,
                      n_jobs_br=1, n_jobs_gs=1, random_state=42, taxon_id=9606,
                      verbose=False, selection=DEFAULT_SELECTION):
    """Calls :func:`paper_model` and trains the returned model on
    all interaction instances that are strictly training (`is_training` flag
    is True) as returned by :func:`load_training_dataset`.

    Parameters:
    ----------
    rcv_splits : int, optional, default: 3
        The number of splits to use during hyper-parameter cross-validation.

    rcv_iter : int, optional, default: 30
        The number of grid search iterations to perform.

    scoring : str, optional default: 'f1'
        Scoring method used during hyperparameter search.

    n_jobs_model : int, optional, default: 1
        Sets the `n_jobs` parameter of the Pipeline's estimator step.

    n_jobs_br : {int}, optional
        Sets the `n_jobs` parameter of the :class:`MixedBinaryRelevanceClassifier`
        step.

    n_jobs_gs : int, optional, default: 1
        Sets the `n_jobs` parameter of the `RandomizedGridSearch` classifier.

    random_state : int or None, optional, default: None
        This is a seed used to generate random_states for all estimator
        objects such as the base model and the grid search.

    taxon_id : int, optional, default: 9606
        Load the training data for a specific organism. Removes all
        training interactions that do not match this value. Ignored if None.

    verbose : bool, optional, default: False
        Log intermediate messages during training.

    selection : list
        List of annotations to use. Select from 'go_mf', 'go_cc', 'go_bp',
        'ulca_go_mf', 'ulca_go_cc', 'ulca_go_bp', 'interpro', 'pfam'.

    Returns
    -------
    tuple
        First element is a :class:`MixedBinaryRelevanceClassifier` and the
        second is a list specifying the feature selection it was trained on.
    """
    dataset = load_training_dataset(taxon_id=taxon_id, selection=selection)
    if not dataset:
        raise ValueError(
            "No training data could be found. "
            "Have you executed the build_data script yet?"
        )

    X, y = dataset.get("training")
    labels = dataset.get("labels")
    clf = paper_model(
        labels=labels, rcv_splits=rcv_splits, rcv_iter=rcv_iter,
        scoring=scoring, n_jobs_gs=n_jobs_gs, n_jobs_br=n_jobs_br,
        random_state=random_state, n_jobs_model=n_jobs_model
    )
    clf.fit(X, y)
    return clf, selection


def save_to_arff(file_path, interactions, labels, selection,
                 vectorizer=None, unlabelled=False, meka=True, use_bzip=True):
    """Formats interactions for `Weka/Meka` by contructing an `arff` file.

    Parameters:
    ----------
    file_path : str
        File path to save the arff file to.

    interactions : list
        List of :class:`Interaction` instances.

    labels : list, optional, default: None
        A list of labels that appear in the interactions. If supplied, the
        labels are sorted and used to initialise a :class:`MultiLabelBinarizer`,
        which is then used to transform the text labels of the interactions
        into binary `indicator-matrix` format.

    selection : list
        List of annotations to use. Select from 'go_mf', 'go_cc', 'go_bp',
        'ulca_go_mf', 'ulca_go_cc', 'ulca_go_bp', 'interpro', 'pfam'.

    vectorizer : :class:`CountVectorizer` or None, optional, default: None
        Supply a vectorizer instance to transform the textual features
        of the interactions. If None, builds a CountVectorizer with
        `lowercase` set to `False` and `binary` set to `True`.

    unlabelled : {bool}, optional
        If True, treats the input dataset as unlablled. Necessary for
        Meka/Weka to make predictions.

    meka : {bool}, optional
        If True, formats the arff file for `Meka`. Otherwise it is formated
        for `Weka`.

    use_bzip : {bool}, optional
        If True, compress the output file using `bzip`, other use `gzip`.

    Returns
    -------
    `None`
    """
    if use_bzip:
        zipper = bz2
    else:
        zipper = gzip

    if vectorizer is None:
        vectorizer = CountVectorizer(lowercase=False, binary=True)

    X, y = interactions_to_Xy_format(interactions, selection)
    mlb = MultiLabelBinarizer(classes=sorted(labels), sparse_output=False)
    if not unlabelled:
        y = mlb.fit_transform(y)
    X = vectorizer.fit_transform(X)

    if meka:
        header = "@relation 'PTMs: -C %d'\n\n" % (len(labels))
    else:
        header = "@relation PTMs\n\n"

    for label in labels:
        header += "@attribute %s {0,1}\n" % (label)
    for feature in (rename(x) for x in vectorizer.get_feature_names()):
        header += "@attribute %s numeric\n" % (feature)

    header += "\n@data\n\n"

    with zipper.open(file_path, 'wb') as fp:
        X = X.todense()
        if unlabelled:
            X = X.astype(str)
            y = y.astype(str)
            y[:, :] = '?'
        vec = np.hstack([y, X])
        np.savetxt(
            fp, X=vec, fmt='%s', delimiter=',', comments='', header=header
        )
