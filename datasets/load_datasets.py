import numpy as np
from torch.utils.data import Dataset as TorchDataset
import torch

class AudioDataset(TorchDataset):
  def __init__(self, mode="train"):

    
    if mode not in ["train", "val"]:
      raise ValueError("mode must be 'train' or 'val'")
    
    self.mode = mode
    if self.mode == "train":
      self.data = [pair.replace('\n','') for pair in open('datasets/training.tsv', 'r').readlines()]
    elif self.mode == "val":
      self.data = [pair.replace('\n','') for pair in open('datasets/valid.tsv', 'r').readlines()]


  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    jp_path, en_path = self.data[idx].split('\t')
    
    jp_audio = np.load(jp_path)
    en_audio = np.load(en_path)
    jp_audio = torch.tensor(jp_audio)
    en_audio = torch.tensor(en_audio)
    
    return jp_audio, en_audio


  



