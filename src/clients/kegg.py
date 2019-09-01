import copy
import io
import logging
from collections import OrderedDict
from itertools import product
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from bs4 import BeautifulSoup

from ..settings import LOGGER_NAME


__all__ = ["EntryID", "Entries", "Entry", "Relation", "Kegg", "KeggPathway"]

logger = logging.getLogger(LOGGER_NAME)

EntryID = str
Entries = Dict[EntryID, "Entry"]


class Entry:
    def __init__(self, eid: str, accessions: List[str], entry_type: str):
        if isinstance(accessions, str):
            accessions = accessions.split(" ")
        self.eid = eid
        self.accessions = accessions
        self.entry_type = entry_type

    def __repr__(self) -> str:
        return str(
            dict(
                eid=self.eid,
                accessions=self.accessions,
                entry_type=self.entry_type,
            )
        )


class Relation:
    def __init__(
        self, source: Entry, target: Entry, category: str, labels: List[str]
    ):
        self.source: Entry = source
        self.target: Entry = target
        self.category: str = category
        self.labels: List[str] = labels

    def __repr__(self) -> str:
        return str(
            dict(
                source=self.source.eid,
                target=self.target.eid,
                category=self.category,
                labels=self.labels,
            )
        )


class KeggPathway:
    def __init__(self, xml: str):
        self.root: BeautifulSoup = BeautifulSoup(xml, features="xml")

    def __repr__(self) -> str:
        return str(self.root)

    @property
    def entries(self) -> Entries:
        entries: Entries = {}
        for entry in self.root.find_all("entry"):
            kegg_id = entry["id"]
            accessions = entry["name"].split(" ")
            entry_type = entry["type"]
            entry = Entry(
                eid=kegg_id, accessions=accessions, entry_type=entry_type
            )
            entries[kegg_id] = entry
        return entries

    @property
    def relations(self) -> List[Relation]:
        relations: List[Relation] = []
        entries: Entries = self.entries
        for relation in self.root.find_all("relation"):
            subtypes: List[Dict] = relation.find_all("subtype")
            source: Entry = entries[relation["entry1"]]
            target: Entry = entries[relation["entry2"]]
            category: str = relation["type"]
            labels: List[str] = [st["name"] for st in subtypes]
            relations.append(
                Relation(
                    source=source,
                    target=target,
                    category=category,
                    labels=labels,
                )
            )
        return relations

    @property
    def interactions(self) -> pd.DataFrame:
        data: Dict[str, List[Any]] = {"source": [], "target": [], "label": []}
        include_categories = ("pprel",)
        for relation in self.relations:
            if relation.category.lower() not in include_categories:
                continue
            if relation.source.entry_type.lower() not in ("gene",):
                continue
            if relation.target.entry_type.lower() not in ("gene",):
                continue

            combinations = product(
                relation.source.accessions, relation.target.accessions
            )
            for s, t in combinations:
                for label in relation.labels:
                    data["source"].append(s)
                    data["target"].append(t)
                    data["label"].append(label)

        return (
            pd.DataFrame(data=data, columns=["source", "target", "label"])
            .drop_duplicates(subset=None, keep="first", inplace=False)
            .dropna(axis=0, how="any", inplace=False)
        )


class Kegg:
    """
    Simple wrapper for KEGG's API.

    Attributes
    ----------
        organisms : str
            KEGG three letter organism code.
    """

    BASE_URL: str = "http://rest.kegg.jp/"

    def __init__(self):
        self.cache: Dict[str, Any] = {}

    @classmethod
    def url_builder(cls, operation: str, arguments: Union[str, List[str]]):
        if isinstance(arguments, str):
            arguments = [arguments]
        if not arguments:
            raise ValueError("At least one argument is required.")
        return "{}{}/{}/".format(cls.BASE_URL, operation, "/".join(arguments))

    @staticmethod
    def get(url: str) -> requests.Response:
        response: requests.Response = requests.get(url)
        if not response.ok:
            logger.error(f"{response.content.decode()}")
            response.raise_for_status()
        return response

    @staticmethod
    def parse_json(response: requests.Response):
        # Raise error if response failed. Only raised for error codes.
        response.raise_for_status()
        return response.json()

    @staticmethod
    def parse_dataframe(
        response: requests.Response,
        delimiter: str = "\t",
        header: Optional[List[int]] = None,
        columns: Optional[List[str]] = None,
    ):
        # Raise error if response failed. Only raised for error codes.
        response.raise_for_status()
        handle = io.StringIO(response.content.decode())

        # Use header to specify column names if defined. Otherwise use
        # user-specifed columns.
        if header:
            return pd.read_csv(handle, delimiter=delimiter, header=header)
        elif columns:
            return pd.read_csv(handle, delimiter=delimiter, names=columns)
        else:
            return pd.read_csv(handle, delimiter=delimiter, header=None)

    @property
    def organisms(self) -> pd.DataFrame:
        url = self.url_builder("list", "organism")

        if url in self.cache:
            return self.cache[url]

        organisms: pd.DataFrame = self.parse_dataframe(
            self.get(url), columns=["accession", "code", "name", "taxonomy"]
        )
        self.cache[url] = organisms
        return organisms

    def pathways(self, organism: str) -> pd.DataFrame:
        url: str = self.url_builder("list", ["pathway", organism])

        if url in self.cache:
            return self.cache[url]

        pathways: pd.DataFrame = self.parse_dataframe(
            self.get(url), columns=["accession", "name"]
        )
        self.cache[url] = pathways
        return pathways

    def genes(self, organism: str) -> pd.DataFrame:
        url = self.url_builder("list", organism)

        if url in self.cache:
            return self.cache[url]

        genes: pd.DataFrame = self.parse_dataframe(
            self.get(url), columns=["accession", "names"]
        )
        self.cache[url] = genes
        return genes

    def gene_detail(self, accession: str) -> str:
        url = self.url_builder("get", accession)

        if url in self.cache:
            return self.cache[url]

        detail: str = self.get(url).content.decode()
        self.cache[url] = detail
        return detail

    def pathway_detail(self, accession: str) -> KeggPathway:
        # Remove 'path:' prefix if present.
        url = self.url_builder("get", [accession.split(":")[-1], "kgml"])

        if url in self.cache:
            return self.cache[url]

        pathway: KeggPathway = KeggPathway(xml=self.get(url).content.decode())
        self.cache[url] = pathway
        return pathway

    def convert(
        self, source: str = "hsa", destination: str = "uniprot"
    ) -> Dict[str, List[str]]:
        url = self.url_builder("conv", [source, destination])

        if url in self.cache:
            return self.cache[url]

        df: pd.DataFrame = self.parse_dataframe(self.get(url))
        mapping: Dict[str, List[str]] = {}
        for row in df.to_dict("records"):
            # KEGG API is structured as destination first then source.
            dst, src = row.values()
            if source != "hsa":
                src = src.split(":")[-1]
            if destination != "hsa":
                dst = dst.split(":")[-1]

            if src in mapping:
                mapping[src].append(dst)
            else:
                mapping[src] = [dst]

        self.cache[url] = mapping

        return mapping
