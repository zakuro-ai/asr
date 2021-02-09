docker rmi -f jcadic/vanilla:deepspeech
docker rmi -f jcadic/deepspeech
docker build . -t jcadic/deepspeech
docker run --rm --gpus all -it  --shm-size=70g  -v /mnt/.cdata:/mnt/.cdata jcadic/deepspeech bash
