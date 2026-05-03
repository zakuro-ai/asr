import importlib
import pkgutil

import asr_deepspeech


def test_version():
    assert asr_deepspeech.__version__


def test_submodule_imports():
    package = asr_deepspeech
    failed = []
    for importer, modname, ispkg in pkgutil.walk_packages(
        path=package.__path__, prefix=package.__name__ + ".", onerror=lambda _: None
    ):
        try:
            importlib.import_module(modname)
        except Exception as exc:
            failed.append((modname, str(exc)))
    assert not failed, f"Failed imports: {failed}"
