import torch
import numpy as np
from typing import Callable

from models.fatchord_version import WaveRNN
from models.forward_tacotron import ForwardTacotron
from utils.dsp import DSP
from utils.text.cleaners import Cleaner
from utils.text.tokenizer import Tokenizer


class Synthesizer:

    def __init__(self,
                 tts_path: str,
                 voc_path: str,
                 device='cuda'):
        self.device = torch.device(device)
        tts_checkpoint = torch.load(tts_path, map_location=self.device)
        tts_config = tts_checkpoint['config']
        tts_model = ForwardTacotron.from_config(tts_config)
        tts_model.load_state_dict(tts_checkpoint['model'])
        self.tts_model = tts_model
        self.wavernn = WaveRNN.from_checkpoint(voc_path)
        try:
            self.melgan = torch.hub.load('seungwonpark/melgan', 'melgan').cuda().eval()
        except Exception as e:
            print(e)
        self.cleaner = Cleaner.from_config(tts_config)
        self.tokenizer = Tokenizer()
        self.dsp = DSP.from_config(tts_config)

    def __call__(self,
                 text: str,
                 voc_model: str,
                 alpha=1.0,
                 pitch_function: Callable[[torch.tensor], torch.tensor] = lambda x: x) -> np.array:
        x = self.cleaner(text)
        x = self.tokenizer(x)
        x = torch.tensor(x).unsqueeze(0)
        _, m, _, _ = self.tts_model.generate(x, alpha=alpha, pitch_function=pitch_function)
        if voc_model == 'griffinlim':
            wav = self.dsp.griffinlim(m, n_iter=32)
        elif voc_model == 'wavernn':
            m = torch.tensor(m).unsqueeze(0)
            wav = self.wavernn.generate(mels=m,
                                        batched=True,
                                        target=11_000,
                                        overlap=550,
                                        mu_law=self.dsp.mu_law)
        else:
            m = torch.tensor(m).unsqueeze(0).cuda()
            with torch.no_grad():
                wav = self.melgan.inference(m).cpu().numpy()
        return wav
