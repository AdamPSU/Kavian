import pandas as pd

def _make_row_value_pair(row, value):
    """This helper returns a (row, value) tuple to be used in
    summary creation."""

    value = str(round(value, 3))
    return f"{row} :", value




