"""JAXBench workload discovery and loading."""

import os

BENCHMARK_DIR = os.path.dirname(os.path.abspath(__file__))


def _search_dirs():
    """The benchmark root plus any nested level* directories (level1/, level2/, ...), so
    workloads stored under a level directory are still discoverable by bare name."""
    dirs = [BENCHMARK_DIR]
    for d in sorted(os.listdir(BENCHMARK_DIR)):
        full = os.path.join(BENCHMARK_DIR, d)
        if os.path.isdir(full) and d.startswith('level'):
            dirs.append(full)
    return dirs


def list_workloads():
    """Return sorted list of workload names (e.g. ['1p_Flash_Attention', ...]).

    Searches the benchmark root and each nested level* directory.
    """
    workloads = []
    for base in _search_dirs():
        for d in os.listdir(base):
            full = os.path.join(base, d)
            if (os.path.isdir(full)
                    and not d.startswith(('_', '.'))
                    and os.path.exists(os.path.join(full, 'baseline.py'))):
                workloads.append(d)
    return sorted(set(workloads), key=_sort_key)


def get_workload_dir(name):
    """Return absolute path to a workload directory, searching the benchmark root and each
    nested level* directory. Raises ValueError if not found."""
    for base in _search_dirs():
        path = os.path.join(base, name)
        if os.path.isdir(path) and os.path.exists(os.path.join(path, 'baseline.py')):
            return path
    raise ValueError(f"Workload '{name}' not found under {BENCHMARK_DIR} (searched the root and level* dirs)")


def has_optimized(name):
    """Check if a workload has an optimized.py variant."""
    try:
        return os.path.exists(os.path.join(get_workload_dir(name), 'optimized.py'))
    except ValueError:
        return False


def _sort_key(name):
    """Sort by numeric prefix (1p before 18k, etc.)."""
    num = ''
    for c in name:
        if c.isdigit():
            num += c
        else:
            break
    return int(num) if num else 999
