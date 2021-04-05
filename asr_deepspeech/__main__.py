from asr_deepspeech.modules import DeepSpeech
from asr_deepspeech import load_config

if __name__ == '__main__':
    config = load_config()
    config.model["model_path"]=None
    model = DeepSpeech(**config.model)
    config.inference["cuda"] = False
    wer, cer, _ = model(**config.inference)
    print(wer, cer)

