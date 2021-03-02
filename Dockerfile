FROM jcadic/vanilla:deepspeech

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt

COPY . /deepspeech
WORKDIR /deepspeech
RUN python setup.py install

CMD ["python", "-m", "asr_deepspeech.test"]
