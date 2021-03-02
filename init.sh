docker rmi -f jcadic/vanilla:deepspeech
docker rmi -f jcadic/deepspeech
docker build . -t jcadic/deepspeech
docker run \
  --rm \
  --gpus all \
  -d \
  --shm-size=70g \
  -v /mnt/.cdata:/mnt/.cdata \
  -v $(pwd)/data/models:/deepspeech/data/models \
  jcadic/deepspeech \
  python -m asr_deepspeech.trainers --batch-size 50
