#docker rmi -f jmcadic/vanilla:deepspeech
#docker rmi -f jmcadic/deepspeech
#docker build . -t jmcadic/deepspeech
docker run \
  --rm \
  --gpus all \
  -it \
  --shm-size=70g \
  -v $(pwd):/workspace \
  -v /srv/sync/:/srv/sync \
  jmcadic/deepspeech bash
#  python -m asr_deepspeech.trainers --batch-size 150
#  python -m asr_deepspeech.trainers  --labels __data__/labels/labels.json --manifest google --batch-size 500
