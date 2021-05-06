import argparse
import itertools
import os
from pathlib import Path

import torch
from torch import optim
from torch.utils.data.dataloader import DataLoader

from models.forward_tacotron import ForwardTacotron
from models.tacotron import Tacotron
from trainer.forward_trainer import ForwardTrainer
from utils.checkpoints import restore_checkpoint
from utils.dataset import get_tts_datasets
from utils.display import *
from utils.files import read_config
from utils.paths import Paths
from utils.text.symbols import phonemes


def create_gta_features(model: Tacotron,
                        train_set: DataLoader,
                        val_set: DataLoader,
                        save_path: Path) -> None:
    model.eval()
    device = next(model.parameters()).device  # use same device as model parameters
    iters = len(train_set) + len(val_set)
    dataset = itertools.chain(train_set, val_set)
    for i, (x, mels, ids, x_lens, mel_lens, dur, pitch) in enumerate(dataset, 1):
        x, m, dur, x_lens, mel_lens, pitch = x.to(device), mels.to(device), dur.to(device), \
                                             x_lens.to(device), mel_lens.to(device), pitch.to(device)

        with torch.no_grad():
            _, gta, _, _ = model(x, mels, dur, mel_lens, pitch)
        gta = gta.cpu().numpy()
        for j, item_id in enumerate(ids):
            mel = gta[j][:, :mel_lens[j]]
            np.save(str(save_path/f'{item_id}.npy'), mel, allow_pickle=False)
        bar = progbar(i, iters)
        msg = f'{bar} {i}/{iters} Batches '
        stream(msg)


if __name__ == '__main__':
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train Tacotron TTS')
    parser.add_argument('--force_gta', '-g', action='store_true', help='Force the model to create GTA features')
    parser.add_argument('--config', metavar='FILE', default='config.yaml', help='The config containing all hyperparams.')
    args = parser.parse_args()

    config = read_config(args.config)
    paths = Paths(config['data_path'], config['voc_model_id'], config['tts_model_id'])

    assert len(os.listdir(paths.alg)) > 0, f'Could not find alignment files in {paths.alg}, please predict ' \
                                           f'alignments first with python train_tacotron.py --force_align!'

    force_gta = args.force_gta
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print('Using device:', device)

    # Instantiate Forward TTS Model
    print('\nInitialising Forward TTS Model...\n')
    model = ForwardTacotron.from_config(config).to(device)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'num params {params}')

    optimizer = optim.Adam(model.parameters())
    restore_checkpoint('forward', paths, model, optimizer, create_if_missing=True)

    if force_gta:
        print('Creating Ground Truth Aligned Dataset...\n')
        train_set, val_set = get_tts_datasets(paths.data, 8, r=1, model_type='forward')
        create_gta_features(model, train_set, val_set, paths.gta)
        print('\n\nYou can now train WaveRNN on GTA features - use python train_wavernn.py --gta\n')
    else:
        trainer = ForwardTrainer(paths)
        trainer.train(model, optimizer)

