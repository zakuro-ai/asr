FROM zakuroai/asr_deepspeech:vanilla

COPY dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl

RUN pip install --no-cache-dir jupyterlab

RUN useradd -m -u 1000 foo

USER foo
WORKDIR /workspace
