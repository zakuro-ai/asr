FROM  jcadic/vanilla:ml

LABEL maintainer="Cadic Jean Maximilien <info@cadic.jp>"
LABEL description="This is a bare-bones example of the asr_deepspeech framework setup."

COPY . /asr_deepspeech
WORKDIR /asr_deepspeech
RUN python setup.py install

WORKDIR /

RUN rm -r /asr_deepspeech

CMD ["python", "-m", "asr_deepspeech.test"]
