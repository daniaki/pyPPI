import io
import gzip
import json
import logging
from collections import OrderedDict

import requests
from csv import DictReader
from requests import Response
from collections import defaultdict
from typing import Union, Optional, List, Dict, Set, Iterable, Any

from bs4 import BeautifulSoup

from ..utilities import is_null
from ..constants import GeneOntologyCategory, Paths
from ..parsers import types
from ..settings import LOGGER_NAME


__all__ = ["UniprotClient", "UniprotEntry"]


logger = logging.getLogger(LOGGER_NAME)


class UniprotEntry:
    def __init__(self, root: BeautifulSoup):
        self.root: BeautifulSoup = root
        if self.root.find("entry"):
            self.root = self.root.find("entry")

    def __str__(self):
        return self.primary_accession

    @property
    def keywords(self) -> List[types.KeywordTermData]:
        kws = []
        for kw in self.root.find_all("keyword", recursive=False):
            kws.append(
                types.KeywordTermData(
                    identifier=kw["id"].upper(), name=kw.text.strip() or None
                )
            )
        return kws

    @property
    def go_terms(self) -> List[types.GeneOntologyTermData]:
        terms = []
        for term in self.root.find_all(
            "dbReference", type="GO", recursive=False
        ):
            details = term.find_all("property", type="term")
            if not details:
                continue

            terms.append(
                types.GeneOntologyTermData(
                    identifier=term["id"].upper(),
                    category=GeneOntologyCategory.letter_to_category(
                        details[0]["value"][0].strip()
                    ),
                    name=details[0]["value"][2:].strip(),
                    obsolete=False,
                )
            )
        return terms

    @property
    def pfam_terms(self) -> List[types.PfamTermData]:
        terms = []
        for term in self.root.find_all(
            "dbReference", type="Pfam", recursive=False
        ):
            details = term.find_all("property", type="entry name")
            if not details:
                continue

            terms.append(
                types.PfamTermData(
                    identifier=term["id"].upper(),
                    name=details[0]["value"].strip(),
                )
            )
        return terms

    @property
    def interpro_terms(self) -> List[types.InterproTermData]:
        terms = []
        for term in self.root.find_all(
            "dbReference", type="InterPro", recursive=False
        ):
            details = term.find_all("property", type="entry name")
            if not details:
                continue
            terms.append(
                types.InterproTermData(
                    identifier=term["id"].upper(),
                    name=details[0]["value"].strip(),
                )
            )
        return terms

    @property
    def sequence(self) -> Optional[str]:
        node = self.root.find("sequence", recursive=False)
        if not node:
            return None
        return node.text.replace("\n", "")

    @property
    def function(self) -> List[str]:
        comments = []
        for elem in self.root.find_all(
            "comment", type="function", recursive=False
        ):
            comments.append(elem.text.strip())
        return comments

    @property
    def accessions(self) -> List[str]:
        return [
            elem.text.upper()
            for elem in self.root.find_all("accession", recursive=False)
        ]

    @property
    def primary_accession(self) -> str:
        return self.accessions[0]

    @property
    def alias_accessions(self) -> List[str]:
        return [
            elem for elem in self.accessions if elem != self.primary_accession
        ]

    @property
    def name(self) -> Optional[str]:
        node = self.root.find("name", recursive=False)
        if not node:
            return None
        return node.text

    @property
    def full_name(self) -> Optional[str]:
        protein = self.root.find("protein", recursive=False)
        if not protein:
            return None
        names = protein.find("recommendedName", recusrive=False)
        if names and names.find("fullName", recusrive=False):
            return names.find("fullName").text
        return None

    @property
    def short_name(self) -> Optional[str]:
        protein = self.root.find("protein", recursive=False)
        if not protein:
            return None
        names = protein.find("recommendedName", recusrive=False)
        if names and names.find("shortName", recusrive=False):
            return names.find("shortName").text
        return None

    @property
    def alt_names(self) -> List[str]:
        protein = self.root.find("protein", recursive=False)
        if not protein:
            return []
        # Find direct children of the protein tag with tag 'alternativeName'
        alt_names = protein.find_all("alternativeName", recursive=False)
        return [name.text.strip() for name in alt_names]

    @property
    def db(self) -> str:
        return self.root["dataset"]

    @property
    def reviewed(self) -> bool:
        return str(self.db).lower() == "swiss-prot"

    @property
    def version(self) -> str:
        return self.root["version"]

    @property
    def genes(self) -> List[types.GeneData]:
        genes: List[types.GeneData] = []
        node = self.root.find("gene", recursive=False)
        if not node:
            return genes
        for gene in node.find_all("name"):
            genes.append(
                types.GeneData(
                    symbol=gene.text.upper(), relation=gene["type"].lower()
                )
            )
        return genes

    @property
    def primary_gene(self) -> Optional[types.GeneData]:
        primary = [gene for gene in self.genes if gene.relation == "primary"]
        return None if not primary else primary[0]

    @property
    def synonym_genes(self) -> List[types.GeneData]:
        return [gene for gene in self.genes if gene.relation == "synonym"]

    @property
    def taxonomy(self) -> Optional[int]:
        organism = self.root.find("organism", recursive=False)
        if not organism:
            return None
        node = organism.find("dbReference", type="NCBI Taxonomy")
        if not node:
            return None
        return int(node["id"])


class UniprotClient:
    def __init__(
        self, api_key: Optional[str] = None, use_cache: Optional[bool] = False
    ):
        self.api_key: Optional[str] = api_key
        self.base: str = "https://www.uniprot.org"
        self.use_cache = use_cache
        self.cache: Dict[str, Any] = {}
        if self.use_cache:
            self._load_cache()

    def _delete_cache(self):
        self.cache = {}
        self._save_cache()

    def _load_cache(self):
        if Paths.uniprot_cache.exists():
            self.cache = json.load(gzip.open(Paths.uniprot_cache, "rt"))

    def _save_cache(self, overwrite: Optional[bool] = False):
        if not overwrite:
            json.dump(self.cache, gzip.open(Paths.uniprot_cache, "wt"))
        else:
            existing = (
                json.load(gzip.open(Paths.uniprot_cache, "rt"))
                if Paths.uniprot_cache.exists()
                else {}
            )
            # Updates existing cache instead of over-writting
            existing.update(self.cache)
            json.dump(existing, gzip.open(Paths.uniprot_cache, "wt"))

    def get_entry(self, identifier: str) -> UniprotEntry:
        url: str = f"{self.base}/uniprot/{identifier}.xml"
        cache_key = f"get-entry:{identifier}"

        if cache_key in self.cache:
            response_data = self.cache[cache_key]
        else:
            response: Response = requests.get(url)
            if not response.ok:
                logger.error(f"{response.content.decode()}")
                response.raise_for_status()
            response_data = response.text
            self.cache[cache_key] = response_data
            self._save_cache()

        return UniprotEntry(BeautifulSoup(response_data, "xml"))

    def get_accession_map(
        self, identifiers: Iterable[str], fr: str = "ACC+ID", to: str = "ACC"
    ) -> Dict[str, List[str]]:
        url = f"{self.base}/uploadlists/"
        cache_key = f"get-map:{hash(sorted(identifiers))}"

        if cache_key in self.cache:
            return self.cache[cache_key]

        data = {
            "query": "\n".join(list(identifiers)),
            "format": "tab",
            "from": fr,
            "to": to,
        }
        response: Response = requests.post(url, data)
        if not response.ok:
            logger.error(f"{response.content.decode()}")
            response.raise_for_status()

        reader = DictReader(io.StringIO(response.text), delimiter="\t")
        mapping: Dict[str, List[str]] = defaultdict(list)
        for row in reader:
            key = row["From"]
            mapping[key] += [
                x.strip() for x in row["To"].split(",") if not is_null(x)
            ]

        # Return all unique values that a key maps to. Must preserve order
        # that values are first encountered in. Using a set will not
        # preserve this ordering.
        result: Dict[str, List[str]] = {
            k: list(OrderedDict.fromkeys(v).keys())
            for (k, v) in mapping.items()
        }

        # Set and save cache
        self.cache[cache_key] = result
        self._save_cache()

        return result

    def get_entries(self, identifiers: Iterable[str]) -> List[UniprotEntry]:
        url = f"{self.base}/uploadlists/"

        # Can take up to 10-20 minutes for large lists so cache result.
        response: Response
        cache_key = f"get-entries:{hash(sorted(identifiers))}"

        if cache_key in self.cache:
            response_data = self.cache[cache_key]
        else:
            data = {
                "query": "\n".join(sorted(identifiers)),
                "format": "xml",
                "from": "ACC+ID",
                "to": "ACC",
            }
            response = requests.post(url, data)
            if not response.ok:
                logger.error(f"{response.content.decode()}")
                response.raise_for_status()
            response_data = response.text

            # Set and save cache
            self.cache[cache_key] = response_data
            self._save_cache()

        return [
            UniprotEntry(entry)
            for entry in BeautifulSoup(response_data, "xml").find_all("entry")
        ]
