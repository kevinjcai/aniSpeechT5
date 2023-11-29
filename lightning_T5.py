from pytorch_lightning import LightningModule, Trainer
from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech

from datasets.load_datasets import AudioDataset

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np



class SpeechT5Module(LightningModule):
    def __init__(self):
        super().__init__()
        
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
        for param in self.model.parameters():
            if not param.requires_grad:
                param.requires_grad = True
        self.loss_function = torch.nn.CrossEntropyLoss()

        # print(self.model.config)
        # raise Exception('stop')
    
    def forward(self, jp_audio, en_audio):

        features = self.processor(audio=[
            waveform.squeeze().cpu().numpy() for waveform in jp_audio
        ], audio_target=[
            waveform1.squeeze().cpu().numpy() for waveform1 in en_audio
        ], return_tensors="pt", padding=True, sampling_rate=16000)

        for i in features:
            features[i] = features[i].to(self.device)
        outputs = self.model(**features)
        return outputs

    def training_step(self, batch, batch_idx):

        jp_audio, en_audio = batch
        output = self(jp_audio, en_audio)
        
        return output

    def validation_step(self, batch, batch_idx):
        jp_audio, en_audio = batch
        
        output = self(jp_audio, en_audio)
        
        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    
    def custom_collate_fn(self, batch):
        jp_audio_tensors = [item[0] for item in batch]
        en_audio_tensors = [item[1] for item in batch]

        jp_audio_padded = pad_sequence(jp_audio_tensors, batch_first=True, padding_value=0.0)
        en_audio_padded = pad_sequence(en_audio_tensors, batch_first=True, padding_value=0.0)

        return jp_audio_padded, en_audio_padded
    
    def train_dataloader(self):
        train_data = AudioDataset(mode='train')
        return DataLoader(train_data, batch_size=2, shuffle=True, collate_fn=self.custom_collate_fn)

    def val_dataloader(self):
        val_data = AudioDataset(mode='val')
        return DataLoader(val_data, batch_size=2, shuffle=False, collate_fn=self.custom_collate_fn)


speech_t5_module = SpeechT5Module()

trainer = Trainer(max_epochs=10, accelerator='gpu')

trainer.fit(speech_t5_module)
