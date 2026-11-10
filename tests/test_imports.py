"""Import smoke tests for every module in the package.

libemg/_streamers/ is the single highest-churn area in the repository (55 of
the last 200 commits to libemg/) and cannot be tested without the matching
hardware. Importing each module is still worth doing: it catches a stale
import, a half-finished dependency migration, or a syntax error in a vendor
module that nobody has a device for.

This is also a guard on `import libemg` itself, which pulls in the streamers,
the GUI and the environments. A broken module anywhere in that chain makes the
whole library unimportable, regardless of what the user meant to use.
"""

import importlib
import pkgutil

import pytest

import libemg

# Modules that open a window, start an event loop, or otherwise refuse to be
# imported on a headless machine. None are known today; add names here with a
# reason rather than deleting the test if one appears.
UNIMPORTABLE: dict[str, str] = {}


def package_modules(package):
    """Every importable module name under a package, recursively."""
    return sorted(
        module.name
        for module in pkgutil.walk_packages(package.__path__, prefix=package.__name__ + ".")
    )


ALL_MODULES = package_modules(libemg)


def test_the_package_itself_imports():
    assert libemg is not None


def test_module_discovery_found_something():
    """Guard against the parametrize below silently collecting nothing."""
    assert len(ALL_MODULES) > 20, f"only found {len(ALL_MODULES)} modules"


@pytest.mark.parametrize("module_name", ALL_MODULES)
def test_module_imports(module_name):
    """Every module in the package imports cleanly.

    A failure here is usually one of three things: a function that was renamed
    or deleted but is still imported somewhere, a dependency whose API changed
    under a loose version pin, or a module that was never run after an edit.
    """
    if module_name in UNIMPORTABLE:
        pytest.skip(UNIMPORTABLE[module_name])
    importlib.import_module(module_name)


@pytest.mark.parametrize(
    "module_name",
    [
        "libemg.data_handler",
        "libemg.datasets",
        "libemg.emg_predictor",
        "libemg.feature_extractor",
        "libemg.feature_selector",
        "libemg.filtering",
        "libemg.offline_metrics",
        "libemg.shared_memory_manager",
        "libemg.streamers",
        "libemg.utils",
    ],
)
def test_public_module_is_reachable_from_the_package(module_name):
    """The modules users are told to import in the documentation."""
    importlib.import_module(module_name)
