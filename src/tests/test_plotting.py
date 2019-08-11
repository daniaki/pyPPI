import os
import numpy as np
from pathlib import Path

from .. import plotting


class TestPlottingFunctions:
    def setup(self):
        self.paths = []

    def teardown(self):
        for path in self.paths:
            if path.is_file:
                path.unlink()
            assert not path.exists()

    def test_plot_heatmaps_saves_to_path(self):
        path = Path(__file__).parent / "data" / "plot.jpg"
        self.paths.append(path)
        plotting.plot_heatmaps(
            path=self.paths[0],
            labels=["a", "b"],
            correlation_matrix=np.random.random_sample((2, 2)),
            similarity_matrix=np.random.random_sample((2, 2)),
        )
        assert path.is_file

    def test_plot_threshold_curve_saves_to_path(self):
        path = Path(__file__).parent / "data" / "plot.jpg"
        self.paths.append(path)
        plotting.plot_threshold_curve(
            path=self.paths[0],
            thresholds=np.random.random_sample((10,)),
            proportions=np.random.random_sample((10,)),
        )
        assert path.is_file
