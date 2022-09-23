FROM zakuroai/asr_deepspeech:vanilla

COPY dist/*.whl /tmp/
RUN pip install /tmp/*.whl && rm -rf /tmp/*.whl

# For dev purpose
RUN conda install \
    jupyter \
    jupyterlab \
    markupsafe==2.0.1 \
    nb_conda \
    -y 

USER foo
WORKDIR /workspace
