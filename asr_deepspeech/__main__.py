from asr_deepspeech.modules import DeepSpeech
from asr_deepspeech import cfg

if __name__ == "__main__":
    model = DeepSpeech(**vars(cfg.model))
    cfg.inference.cuda = False
    wer, cer, _ = model(**vars(cfg.inference))
    print(wer, cer)
