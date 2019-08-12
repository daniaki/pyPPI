import mock
import gzip
from pathlib import Path

from .. import utilities


null_values = ("none", "na", "nan", "n/a", "undefined", "unknown", "null", " ")


class TestIsNull:
    def test_true_for_null_values(self):
        for value in null_values:
            assert utilities.is_null(value)

    def test_false_for_non_null_values(self):
        assert not utilities.is_null("aaa")


class TestDownloadFromURL:
    def setup(self):
        self.paths = []

    def teardown(self):
        for path in self.paths:
            if path.is_file:
                path.unlink()
            assert not path.exists()

    def test_compresses_with_gzip_if_compress_is_true(self):
        tmp = Path(__file__).parent / "data" / "tmp_download"
        path = Path(__file__).parent / "data" / "response.gz"
        self.paths.append(path)
        self.paths.append(tmp)

        with open(tmp, "wb") as fp:
            fp.write(b"Hello world")

        with mock.patch(
            "urllib.request.urlretrieve", return_value=(tmp, None)
        ) as patch:
            utilities.download_from_url(
                url="aaa", save_path=path, compress=True
            )
            patch.assert_called()
            assert path.exists()
            with gzip.open(path, "rt") as fp:
                assert fp.read() == "Hello world"

    @mock.patch("urllib.request.urlretrieve")
    def test_saves_to_path(self, patch):
        utilities.download_from_url(url="url", save_path="aaa", compress=False)
        patch.assert_called_with(*("url", "aaa"))

