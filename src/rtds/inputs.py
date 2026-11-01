"""Input builder for Part 1: each visit becomes one plain column."""

import pandas as pd

from rtds.interfaces import visit_columns


class FlatVisitColumns:
    def build(self, visits):
        return pd.DataFrame(visits, columns=visit_columns(visits.shape[1]))
