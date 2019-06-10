import logging
import sys
from tqdm import tqdm

from ..constants import Urls, Paths
from ..settings import LOGGER_NAME, PROJECT_NAME
from ..utilities import download_from_url

logger: logging.Logger = logging.getLogger(LOGGER_NAME)


def download_program_data():
    # HPRD must be downloaded manually by the user due to licencing.
    sys.stdout.write(
        "Downloading required data. This may take some time depending on your "
        "network bandwidth.\n\n"
        "NOTE: If you would like to parse additional training samples "
        "from HPRD then you will need to manually download the files from "
        "http://hprd.org/download/ due to licensing constraints. Once "
        "downloaded, move the files '{f1}' and '{f2}' to '{l1}' and '{l2}' "
        "in your home directory (you may need to un-hide hidden "
        "folders).\n\n".format(
            f1="POST_TRANSLATIONAL_MODIFICATIONS.txt",
            f2="HPRD_ID_MAPPINGS.txt",
            l1=Paths.hprd_ptms,
            l2=Paths.hprd_xref,
        )
    )
    attrs = [
        ("interpro_entries", False),
        ("pfam_clans", True),
        ("psimi_obo", True),
        ("go_obo", True),
        ("pina2_mitab", True),
        ("bioplex", True),
        ("innate_all", False),
        ("innate_curated", False),
    ]
    for attr, compress in tqdm(attrs):
        download_from_url(
            url=getattr(Urls, attr),
            save_path=getattr(Paths, attr),
            compress=compress,
        )
