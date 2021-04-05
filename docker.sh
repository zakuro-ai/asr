docker rmi -f jmcadic/deepspeech
docker build . -t jmcadic/deepspeech
docker run \
  --rm \
  --gpus "device=1" \
  -it \
  --shm-size=70g \
  -v $(pwd):/workspace \
  -v /srv/sync/:/srv/sync \
  -v $HOME/.zakuro:/root/.zakuro \
   jmcadic/deepspeech  python -m asr_deepspeech
