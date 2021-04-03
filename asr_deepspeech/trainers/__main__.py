from torch.nn import CTCLoss
from asr_deepspeech.modules import DeepSpeech
from asr_deepspeech.trainers import DeepSpeechTrainer
from asr_deepspeech import load_config

if __name__ == '__main__':
    config = load_config()
    model = DeepSpeech(**config.model)
    trainer = DeepSpeechTrainer(model=model,
                                criterion=CTCLoss(reduction="sum"),
                                **config.trainer)
    trainer.run()

