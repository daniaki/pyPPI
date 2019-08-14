import io
import requests

from csv import DictReader
from requests import Response
from collections import defaultdict
from typing import Union, Optional, List, Dict, Set

from bs4 import BeautifulSoup


class UniprotEntry:
    def __init__(self, root: BeautifulSoup):
        self.root: BeautifulSoup = root

    @property
    def keywords(self) -> List[Dict[str, str]]:
        kws = []
        for kw in self.root.find_all("keyword"):
            kws.append({"id": kw["id"], "text": kw.text})
        return kws

    @property
    def go_terms(self) -> List[Dict[str, str]]:
        terms = []
        for term in self.root.find_all("dbReference", type="GO"):
            details = term.find_all("property", type="term")
            if not details:
                continue
            terms.append(
                {
                    "id": term["id"],
                    "category": details[0]["value"][0],
                    "name": details[0]["value"][2:],
                }
            )
        return terms

    @property
    def pfam_terms(self) -> List[Dict[str, str]]:
        terms = []
        for term in self.root.find_all("dbReference", type="Pfam"):
            details = term.find_all("property", type="entry name")
            if not details:
                continue
            terms.append({"id": term["id"], "name": details[0]["value"]})
        return terms

    @property
    def interpro_terms(self) -> List[Dict[str, str]]:
        terms = []
        for term in self.root.find_all("dbReference", type="InterPro"):
            details = term.find_all("property", type="entry name")
            if not details:
                continue
            terms.append({"id": term["id"], "name": details[0]["value"]})
        return terms

    @property
    def sequence(self) -> Optional[str]:
        node = self.root.find("sequence")
        if not node:
            return None
        return node.text.replace("\n", "")

    @property
    def function(self) -> List[str]:
        comments = []
        for elem in self.root.find_all("comment", type="function"):
            comments.append(elem.text.strip())
        return comments

    @property
    def accessions(self) -> List[str]:
        return [elem.text for elem in self.root.find_all("accession")]

    @property
    def name(self) -> Optional[str]:
        node = self.root.find("name")
        if not node:
            return None
        return node.text

    @property
    def full_name(self) -> Optional[str]:
        names = self.root.find("recommendedName")
        if names and names.find("fullName"):
            return names.find("fullName").text
        return None

    @property
    def short_name(self) -> Optional[str]:
        names = self.root.find("recommendedName")
        if names and names.find("shortName"):
            return names.find("shortName").text
        return None

    @property
    def db(self) -> str:
        return self.root.entry["dataset"]

    @property
    def reviewed(self) -> bool:
        return str(self.db).lower() == "swiss-prot"

    @property
    def version(self) -> str:
        return self.root.entry["version"]

    @property
    def genes(self) -> List[Dict[str, str]]:
        genes: List[Dict[str, str]] = []
        node = self.root.find("gene")
        if not node:
            return genes
        for gene in node.find_all("name"):
            genes.append({"gene": gene.text, "relation": gene["type"]})
        return genes

    @property
    def taxonomy(self) -> Optional[int]:
        organism = self.root.find("organism")
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

    def get_entry(
        self, identifier: str, db: str = "uniprot", fmt: str = "xml"
    ) -> Union[str, UniprotEntry]:
        url: str = f"{self.base}/{db}/{identifier}.{fmt}"
        response: Response = requests.get(url)
        if not response.ok:
            response.raise_for_status()
        if fmt == "xml":
            return UniprotEntry(BeautifulSoup(response.text, "xml"))
        return response.text

    def get_entries(
        self,
        identifiers: str,
        fr: str = "ACC+ID",
        to: str = "ACC",
        fmt: str = "xml",
    ) -> Union[str, List[UniprotEntry], Dict[str, List[str]]]:
        url = f"{self.base}/uploadlists/"
        data = {
            "query": "\n".join(list(identifiers)),
            "format": fmt,
            "from": fr,
            "to": to,
        }
        response: Response = requests.post(url, data)
        if not response.ok:
            response.raise_for_status()
        if fmt == "xml":
            return [
                UniprotEntry(entry)
                for entry in BeautifulSoup(response.text, "xml").find_all(
                    "entry"
                )
            ]
        elif fmt == "tab":
            reader = DictReader(io.StringIO(response.text), delimiter="\t")
            rows: Dict[str, Set[str]] = defaultdict(lambda: set())
            for row in reader:
                map_to: Set[str] = {
                    x.strip() for x in row["To"].split(",") if x.strip()
                }
                rows[row["From"]] |= map_to
            return {k: list(v) for (k, v) in rows.items()}
        return response.text
