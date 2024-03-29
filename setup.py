from setuptools import setup
from asr_deepspeech import __version__

setup(
    name="asr_deepspeech",
    version=__version__,
    short_description="ASRDeepspeech (English / Japanese)",
    long_description="".join(open("README.md", "r").readlines()),
    long_description_content_type="text/markdown",
    url="https://github.com/zakuro-ai/asr",
    license="MIT Licence",
    author="CADIC Jean-Maximilien",
    python_requires=">=3.8",
    packages=[
        "asr_deepspeech",
        "asr_deepspeech.audio",
        "asr_deepspeech.data",
        "asr_deepspeech.data.dataset",
        "asr_deepspeech.data.loaders",
        "asr_deepspeech.data.manifests",
        "asr_deepspeech.data.parsers",
        "asr_deepspeech.data.samplers",
        "asr_deepspeech.decoders",
        "asr_deepspeech.etl",
        "asr_deepspeech.loggers",
        "asr_deepspeech.models",
        "asr_deepspeech.modules",
        "asr_deepspeech.parsers",
        "asr_deepspeech.tests",
        "asr_deepspeech.trainers",
    ],
    include_package_data=True,
    package_data={"": ["*.yml"]},
    install_requires=[r.rsplit()[0] for r in open("requirements.txt")],
    author_email="git@zakuro.ai",
    description="ASRDeepspeech (English / Japanese)",
    platforms="linux_debian_10_x86_64",
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
    ],
)
