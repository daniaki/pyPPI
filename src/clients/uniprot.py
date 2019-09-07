import gzip
import io
import json
import logging
from collections import OrderedDict, defaultdict
from csv import DictReader
from typing import Any, Dict, Iterable, List, Optional, Set, Generator, Tuple

import tqdm
import requests
from bs4 import BeautifulSoup, Tag
from requests import Response
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..constants import GeneOntologyCategory, Paths
from ..parsers import types
from ..settings import LOGGER_NAME
from ..utilities import is_null, chunks


__all__ = ["UniprotClient", "UniprotEntry"]


logger = logging.getLogger(LOGGER_NAME)


class UniprotEntry:
    def __init__(self, root: BeautifulSoup):
        self.root: BeautifulSoup = root
        if not isinstance(self.root, Tag):
            raise TypeError(
                f"Root element must have type 'Tag'. "
                f"Found '{type(self.root).__name__}'."
            )

        if self.root.name != "entry":
            if self.root.find("entry"):
                self.root = self.root.find("entry")

        if self.root.name != "entry":
            raise ValueError(
                f"Root element tag name must be 'entry' . "
                f"Found '{self.root.name}'."
            )

    def __str__(self):
        return self.primary_accession

    @property
    def keywords(self) -> List[types.KeywordTermData]:
        kws = []
        for kw in self.root.find_all("keyword", recursive=False):
            kws.append(
                types.KeywordTermData(
                    identifier=kw["id"].strip().upper(),
                    name=kw.text.strip() or None,
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
                    identifier=term["id"].strip().upper(),
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
                    identifier=term["id"].strip().upper(),
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
                    identifier=term["id"].strip().upper(),
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
            elem.text.strip().upper()
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
    def primary_genes(self) -> List[types.GeneData]:
        return [gene for gene in self.genes if gene.relation == "primary"]

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
        self,
        base: str = "https://www.uniprot.org",
        use_cache: bool = False,
        max_retries: int = 5,
        verbose: bool = False,
    ):
        self.base = base
        self.use_cache = use_cache
        self.cache: Dict[str, Any] = {}
        self.verbose = verbose
        if self.use_cache:
            self._load_cache()

        # Create re-try handler and mount it to the session using the base
        # url as the prefix
        self.session = requests.Session()
        retries = Retry(total=max_retries, respect_retry_after_header=True)
        self.session.mount(self.base, HTTPAdapter(max_retries=retries))

    def _delete_cache(self):
        self.cache = {}
        self._save_cache()

    def _load_cache(self):
        if Paths.uniprot_cache.exists():
            self.cache = json.load(gzip.open(Paths.uniprot_cache, "rt"))

    def _save_cache(self, overwrite: bool = False):
        if not self.use_cache:
            # Bypass saving if use has requested not to use cache.
            return

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

    def _get_entry_from_cache(self, identifier):
        return self.cache.get(identifier, {}).get("entry", None)

    def _get_mapping_from_cache(self, identifier, key):
        return self.cache.get(identifier, {}).get(key, None)

    def _set_entry_in_cache(self, identifier, value):
        if identifier in self.cache:
            self.cache[identifier].update({"entry": value})
        else:
            self.cache.update({identifier: {"entry": value}})

    def _set_mapping_in_cache(self, identifier, key, value):
        if identifier in self.cache:
            self.cache[identifier].update({key: value})
        else:
            self.cache.update({identifier: {key: value}})

    def get_entry(self, identifier: str) -> Optional[UniprotEntry]:
        """
        Download the XML file associated with a UniProt identifier.
        
        Each XML file will be parsed into a `UniProtEntry` instance which
        wraps over a `BeautifulSoup` instance containing the raw 
        XML. This object contains helper utilities to extract a limited number 
        of fields. 
        
        Parameters
        ----------
        identifier : str
            A UniProt protein identifier.
        
        Returns
        -------
        Optional[UniprotEntry]
        """
        url: str = f"{self.base}/uniprot/{identifier.upper()}.xml"
        normalized = identifier.upper()

        data = self._get_entry_from_cache(normalized)
        if data is None:
            response: Response = self.session.get(url)
            if not response.ok:
                if response.status_code == 404:
                    logger.warning(
                        f"Could not find any record for '{normalized}'. "
                        f"Maybe it is obsolete or has been deleted from "
                        f"UniProt?"
                    )
                    return None
                else:
                    logger.error(f"{response.content.decode()}")
                    response.raise_for_status()

            data = response.text
            self._set_entry_in_cache(normalized, data)
            self._save_cache()

        return UniprotEntry(BeautifulSoup(data, "xml"))

    def get_entries(
        self, identifiers: Iterable[str], batch_size: int = 250
    ) -> Generator[Tuple[str, Optional[UniprotEntry]], None, None]:
        """
        Download the XML files associated with an iterable of UniProt 
        identifiers. Each XML file will be parsed into a `UniProtEntry`
        instance which wraps over a `BeautifulSoup` instance containing the raw 
        XML. This object contains helper utilities to extract a limited number 
        of fields. 
        
        Parameters
        ----------
        identifiers : Iterable[str]
            An iterable of UniProt protein identifiers
        batch_size : int, optional
            Request batch size. Setting this too high will result in a 
            connect reset, by default 250
        
        Returns
        -------
        Generator[Tuple[str. Optional[UniprotEntry]]
            Identifiers for which no information could be found will have 
            a `None` value. Returns identifier and associated entry.
        """
        url = f"{self.base}/uploadlists/"
        unique = list(sorted(set(i.upper() for i in identifiers)))

        not_in_cache = []
        for identifier in unique:
            hit = self._get_entry_from_cache(identifier)
            if hit is None:
                not_in_cache.append(identifier)

        batches = list(chunks(not_in_cache, batch_size))
        if self.verbose and batches:
            logger.info(
                f"Requesting {len(batches)} batches of size {batch_size}."
            )
            batches = tqdm.tqdm(batches, total=len(batches))

        for batch in batches:
            data = {
                "query": " ".join(batch),
                "format": "xml",
                "from": "ACC+ID",
                "to": "ACC",
            }
            response: Response = self.session.post(url, data=data)
            if not response.ok:
                logger.error(f"{response.content.decode()}")
                # Save cache before raising.
                self._save_cache()
                response.raise_for_status()

            # Parse the result into separate entries for more modular caching.
            for entry in BeautifulSoup(response.text, "xml").find_all("entry"):
                uniprot_entry = UniprotEntry(entry)
                for identifier in batch:
                    isoform = identifier.split("-")[0]
                    if isoform in uniprot_entry.accessions:
                        self._set_entry_in_cache(
                            identifier, str(uniprot_entry.root)
                        )
                        self._set_entry_in_cache(
                            isoform, str(uniprot_entry.root)
                        )

        # Save updated cache all batches if new items were downloaded.
        if len(not_in_cache):
            self._save_cache()

        if self.verbose:
            logger.info("Parsing XML into UniprotEntry instances.")

        for identifier in unique:
            xml_data = self._get_entry_from_cache(identifier)
            if xml_data is None:
                logger.warning(
                    f"Could not find any record for '{identifier}'. Maybe "
                    f"it is obsolete or has been deleted from UniProt?"
                )
                yield identifier, None
            else:
                yield identifier, UniprotEntry(BeautifulSoup(xml_data, "xml"))

    def get_mapping_table(
        self,
        identifiers: Iterable[str],
        fr: str = "ACC+ID",
        to: str = "ACC",
        batch_size: int = 250,
    ) -> Dict[str, List[str]]:
        """
        Download the a mapping file from a source (`fr`) database to a target 
        (`to`) databse.
        
        Parameters
        ----------
        identifiers : Iterable[str]
            An iterable of UniProt protein identifiers
        batch_size : int, optional
            Request batch size. Setting this too high will result in a 
            connect reset, by default 250
        
        Returns
        -------
        Dict[str, List[str]]
            Identifiers for which no information could be found will have 
            an empty list.
        """
        url = f"{self.base}/uploadlists/"
        unique = list(sorted(set(i.upper() for i in identifiers)))
        mapping_key = f"{fr}-{to}-tab"

        not_in_cache = []
        for identifier in unique:
            hit = self._get_mapping_from_cache(identifier, key=mapping_key)
            if hit is None:
                not_in_cache.append(identifier)

        batches = list(chunks(not_in_cache, batch_size))
        if self.verbose and batches:
            logger.info(
                f"Requesting {len(batches)} batches of size {batch_size}."
            )
            batches = tqdm.tqdm(batches, total=len(batches))

        for batch in batches:
            data = {
                "query": " ".join(batch),
                "format": "tab",
                "from": fr,
                "to": to,
            }
            response: Response = self.session.post(url, data=data)
            if not response.ok:
                logger.error(f"{response.content.decode()}")
                response.raise_for_status()

            mapping: Dict[str, List[str]] = defaultdict(list)
            reader = DictReader(io.StringIO(response.text), delimiter="\t")
            for row in reader:
                key = row["From"]
                # Must preserve order that values are first encountered in.
                # Using a set will not preserve this ordering.
                mapping[key] += [
                    x.strip() for x in row["To"].split(",") if not is_null(x)
                ]

            # Add the mapping values under identifier->mapping_key
            for identifier, values in mapping.items():
                self._set_mapping_in_cache(identifier, mapping_key, values)

        # Save updated cache
        if len(not_in_cache):
            self._save_cache()

        result: Dict[str, List[str]] = {}
        for identifier in unique:
            result[identifier] = (
                self._get_mapping_from_cache(identifier, key=mapping_key) or []
            )

        return result
