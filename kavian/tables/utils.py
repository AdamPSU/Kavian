import numpy as np

def format_stat(stat):
    is_numpy_float = isinstance(stat, np.number) and np.issubdtype(stat.dtype, np.floating)

    if is_numpy_float:
        return f"{stat:.3f}"

    return str(stat)


def format_scientific_notation(stat):
    is_numpy_float = isinstance(stat, np.number) and np.issubdtype(stat.dtype, np.floating)

    if is_numpy_float:
        return f"{stat:.1e}"

    return str(stat)