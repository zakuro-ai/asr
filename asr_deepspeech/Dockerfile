FROM  jcadic/vanilla:torch3.6

LABEL maintainer="Cadic Jean Maximilien <info@cadic.jp>"
LABEL description="This is a bare-bones example of the asr_deepspeech framework setup."

RUN apt update && apt install cmake -y

COPY . /asr_deepspeech
WORKDIR /asr_deepspeech
RUN python setup.py install

WORKDIR /

RUN rm -r /asr_deepspeech

CMD ["python", "-m", "asr_deepspeech.test"]
