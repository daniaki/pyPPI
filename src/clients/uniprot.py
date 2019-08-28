import io
from collections import OrderedDict
import requests

from csv import DictReader
from requests import Response
from collections import defaultdict
from typing import Union, Optional, List, Dict, Set, Iterable

from bs4 import BeautifulSoup

from ..utilities import is_null
from ..constants import GeneOntologyCategory
from ..parsers import types


class UniprotEntry:
    def __init__(self, root: BeautifulSoup):
        self.root: BeautifulSoup = root

    def __str__(self):
        return self.primary_accession

    @property
    def keyword_annotations(self) -> List[types.KeywordTermData]:
        kws = []
        for kw in self.root.find_all("keyword", recursive=False):
            kws.append(
                types.KeywordTermData(
                    identifier=kw["id"].upper(), name=kw.text.strip() or None
                )
            )
        return kws

    @property
    def go_annotations(self) -> List[types.GeneOntologyTermData]:
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
    def pfam_annotations(self) -> List[types.PfamTermData]:
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
    def interpro_annotations(self) -> List[types.InterproTermData]:
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
    def primary_accession(self) -> str:
        return self.accessions[0]

    @property
    def accessions(self) -> List[str]:
        return [
            elem.text
            for elem in self.root.find_all("accession", recursive=False)
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
    def genes(self) -> List[Dict[str, str]]:
        genes: List[Dict[str, str]] = []
        node = self.root.find("gene", recursive=False)
        if not node:
            return genes
        for gene in node.find_all("name"):
            genes.append({"gene": gene.text, "relation": gene["type"]})
        return genes

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
    def __init__(self, api_key: Optional[str] = None):
        self.api_key: Optional[str] = api_key
        self.base: str = "https://www.uniprot.org"

    def get_entry(self, identifier: str) -> UniprotEntry:
        url: str = f"{self.base}/uniprot/{identifier}.xml"
        response: Response = requests.get(url)
        if not response.ok:
            response.raise_for_status()
        return UniprotEntry(BeautifulSoup(response.text, "xml"))

    def get_accession_map(
        self, identifiers: Iterable[str], fr: str = "ACC+ID", to: str = "ACC"
    ) -> Dict[str, List[str]]:
        url = f"{self.base}/uploadlists/"
        data = {
            "query": "\n".join(list(identifiers)),
            "format": "tab",
            "from": fr,
            "to": to,
        }
        response: Response = requests.post(url, data)
        if not response.ok:
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
        return {
            k: list(OrderedDict.fromkeys(v).keys())
            for (k, v) in mapping.items()
        }

    def get_entries(self, identifiers: Iterable[str]) -> List[UniprotEntry]:
        url = f"{self.base}/uploadlists/"
        data = {
            "query": "\n".join(list(identifiers)),
            "format": "xml",
            "from": "ACC+ID",
            "to": "ACC",
        }
        response: Response = requests.post(url, data)
        if not response.ok:
            response.raise_for_status()

        print(response.text)
        return [
            UniprotEntry(entry)
            for entry in BeautifulSoup(response.text, "xml").find_all("entry")
        ]
