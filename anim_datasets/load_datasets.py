import numpy as np
from torch.utils.data import Dataset as TorchDataset
import torch
import librosa
import json
import os

class AudioDataset(TorchDataset):
  def __init__(self, mode="train", path = 'anim_datasets/anim400k_annotations_split_v1.json', base_path = '/data/crunchyroll/audio_clip_relocation/'):
    
    self.base_path = base_path
    self.mode = mode
    
    json_data = json.load(open(path, 'r'))

    if mode not in ["train", "val", "test"]:
      raise ValueError("mode must be 'train' or 'val' or 'test'")

    self.data = json_data[mode]
    
    self.clips = []
    for show in self.data:
      eps = show['episodes']
      for ep_num in eps:
        ep = eps[ep_num]
        for clip in ep['clips']:
          jp_path, en_path = clip['jp_clip'], clip['en_clip']
          jp_path = os.path.join(self.base_path, jp_path + '.mp3')
          en_path = os.path.join(self.base_path, en_path + '.mp3')
          if os.path.exists(jp_path) and os.path.exists(en_path):          
            self.clips.append([jp_path, en_path])
    

  def __len__(self):
    return len(self.clips)

  def __getitem__(self, idx):
    jp_path, en_path = self.clips[idx]

    jp_audio, _ = librosa.load(jp_path, sr=None)
    en_audio, _ = librosa.load(en_path, sr=None)

    jp_audio = torch.tensor(jp_audio)
    en_audio = torch.tensor(en_audio)

    # Limit the length of the audio to 10 seconds
    if jp_audio.shape[0] > 160000:
      jp_audio = jp_audio[:160000]
    if en_audio.shape[0] > 160000:
      en_audio = en_audio[:160000]

    return jp_audio, en_audio
