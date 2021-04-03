from asr_deepspeech import DeepSpeech
from asr_deepspeech import load_config

if __name__ == '__main__':
    config = load_config()
    model = DeepSpeech(**config.model)
    model(**config.inference)
