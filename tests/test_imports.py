import importlib
import pkgutil

import asr_deepspeech

# asr_deepspeech.trainers.__main__ is a training entry point that imports
# optional sakura symbols (asr_metrics / AsyncTrainer) not present in a minimal
# install; it is exercised at runtime, not by the import sweep.
SKIP_MODULES = {
    "asr_deepspeech.trainers.__main__",
}


def test_version():
    assert asr_deepspeech.__version__


def test_submodule_imports():
    package = asr_deepspeech
    failed = []
    for importer, modname, ispkg in pkgutil.walk_packages(
        path=package.__path__, prefix=package.__name__ + ".", onerror=lambda _: None
    ):
        if modname in SKIP_MODULES:
            continue
        try:
            importlib.import_module(modname)
        except Exception as exc:
            failed.append((modname, str(exc)))
    assert not failed, f"Failed imports: {failed}"
