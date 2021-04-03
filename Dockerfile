FROM jmcadic/vanilla:deepspeech

RUN apt install ffmpeg sox -y

COPY requirements.txt /tmp

WORKDIR /tmp

RUN pip install -r requirements.txt

WORKDIR /workspace

CMD ["python", "-m", "asr_deepspeech.test"]

