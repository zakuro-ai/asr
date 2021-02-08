FROM jcadic/vanilla:deepspeech
RUN apt-get install -y locales
RUN sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && \
    locale-gen
ENV LC_ALL en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US.UTF-8
COPY . /deepspeech

WORKDIR /deepspeech

RUN python setup.py install

CMD ["python", "-m", "asr_deepspeech.test"]
