import io
import requests

from csv import DictReader
from requests import Response
from collections import defaultdict
from typing import Union, Optional, List, Dict, Set

from bs4 import BeautifulSoup


class UniprotClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key: Optional[str] = api_key
        self.base: str = "https://www.uniprot.org"

    def get_entry(
        self, identifier: str, db: str = "uniprot", fmt: str = "xml"
    ) -> Union[str, BeautifulSoup]:
        url: str = f"{self.base}/{db}/{identifier}.{fmt}"
        response: Response = requests.get(url)
        if not response.ok:
            response.raise_for_status()
        if fmt == "xml":
            return BeautifulSoup(response.text, "html5lib")
        return response.text

    def get_entries(
        self,
        identifiers: str,
        fr: str = "ACC+ID",
        to: str = "ACC",
        fmt: str = "xml",
    ) -> Union[str, List[BeautifulSoup], Dict[str, List[str]]]:
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
            return BeautifulSoup(response.text, "html5lib").find_all("entry")
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
        for term in self.root.find_all("dbreference", type="GO"):
            terms.append(
                {
                    "id": term["id"],
                    "category": term.property["value"].split(":")[0],
                    "name": term.property["value"].split(":")[1],
                }
            )
        return terms

    @property
    def pfam_terms(self) -> List[Dict[str, str]]:
        terms = []
        for term in self.root.find_all("dbreference", type="Pfam"):
            terms.append({"id": term["id"], "name": term.property["value"]})
        return terms

    @property
    def interpro_terms(self) -> List[Dict[str, str]]:
        terms = []
        for term in self.root.find_all("dbreference", type="InterPro"):
            terms.append({"id": term["id"], "name": term.property["value"]})
        return terms

    @property
    def sequence(self) -> str:
        return self.root.find("sequence").text.replace("\n", "")

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
    def name(self) -> str:
        return self.root.find("name").text

    @property
    def full_name(self) -> str:
        return self.root.find("recommendedname").find("fullname").text

    @property
    def short_name(self) -> str:
        return self.root.find("recommendedname").find("shortname").text

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
        genes = []
        for gene in self.root.find("gene").find_all("name"):
            genes.append({"gene": gene.text, "relation": gene["type"]})
        return genes

    @property
    def taxonomy(self) -> str:
        return self.root.find("organism").find(
            "dbreference", type="NCBI Taxonomy"
        )["id"]
