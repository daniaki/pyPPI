import os
import pathlib

from .. import settings


class TestSettings:
    def setup(self):
        self.path = pathlib.Path(__file__).parent / "data" / "A" / "B"

    def teardown(self):
        if os.path.isdir(self.path):
            self.path.rmdir()
            self.path.parent.rmdir()
        assert not self.path.is_dir()
        assert not self.path.parent.is_dir()

    def test_make_directory_tree(self):
        settings.make_home_dirs([self.path.absolute()])
        assert os.path.isdir(self.path)

    def test_no_error_if_directory_exists(self):
        settings.make_home_dirs([self.path.absolute()])
        assert os.path.isdir(self.path)
        # Make again
        settings.make_home_dirs([self.path.absolute()])
