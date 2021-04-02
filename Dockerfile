FROM jmcadic/vanilla:deepspeech

RUN apt install ffmpeg sox -y

COPY requirements.txt /tmp
WORKDIR /tmp
RUN pip install -r requirements.txt

COPY . /deepspeech
WORKDIR /deepspeech
RUN python setup.py install

CMD ["python", "-m", "asr_deepspeech.test"]
