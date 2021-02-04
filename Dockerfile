FROM jcadic/vanilla:deepspeech

COPY . /deepspeech

WORKDIR /deepspeech

RUN python setup.py install

CMD ["python", "-m", "asr_deepspeech.test"]
