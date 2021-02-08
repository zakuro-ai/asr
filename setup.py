from setuptools import setup
import json

setup(
    name="asr_deepspeech",
    version="1.0a1",
    short_description="ASRDeepspeech (English / Japanese)",
    long_description="ASRDeepspeech (English / Japanese)",
    packages=json.load(open("packages.json", "r")),
    include_package_data=True,
    package_data=json.load(open("package_data.json", "r")),
    url='https://github.com/JeanMaximilienCadic/ASRDeepSpeech',
    license='MIT Licence',
    author='CADIC Jean-Maximilien',
    python_requires='>=3.6',
    install_requires=json.load(open("requirements.json", "r")),
    author_email='info@cadic.jp',
    description='ASRDeepspeech (English / Japanese)',
    platforms="linux_debian_10_x86_64",
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
    ]
)

