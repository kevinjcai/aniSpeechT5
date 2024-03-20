import warnings
from typing import Optional, Tuple, Union
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint

from transformers import SpeechT5Processor
from transformers.models.speecht5.modeling_speecht5 import (
    SpeechT5SpectrogramLoss,
    SpeechT5SpeechDecoderPostnet,
    SpeechT5PreTrainedModel,
    SpeechT5Config,
    SpeechT5DecoderWithSpeechPrenet,
    SpeechT5EncoderWithSpeechPrenet,
    SpeechT5Model,
    Seq2SeqSpectrogramOutput,
    SpeechT5HifiGan,
    shift_spectrograms_right,
    _generate_speech,
)

from diffusers.models.unets.unet_2d import UNet2DModel

from datasets.load_datasets import AudioDataset

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np





class UNetModule(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        self.processor.do_normalize = True
        self.model = UNet2DModel(
            in_channels=1,
            out_channels=1,
        )
        self.hifi_gan = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
        self.loss_fn = SpeechT5SpectrogramLoss(SpeechT5Config())

        for p in self.hifi_gan.parameters():
            p.requires_grad = False

    def forward(self, jp_audio, en_audio):
        features = self.processor(
            audio=[waveform.squeeze().cpu().numpy() for waveform in jp_audio],
            audio_target=[waveform.squeeze().cpu().numpy() for waveform in en_audio],
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
        )
        assert features is not None

        for key, value in features.items():
            features[key] = value.to(self.device)

        outputs = self.model(**features)

        return outputs, features

    def training_step(self, batch, batch_idx):
        jp_audio, en_audio = batch
        output = self(jp_audio, en_audio)

        output, features = output
        features = features["labels"]
        # np.save("holder/features.npy", features.cpu().detach().numpy())
        # np.save("holder/spectrogram.npy", output.spectrogram.cpu().detach().numpy())
        # np.save("holder/waveform.npy", en_audio[0].cpu().detach().numpy())
        # np.save("holder/waveform_jp.npy", jp_audio[0].cpu().detach().numpy())
        # np.save("holder/spec_to_wave.npy",self.hifi_gan(features).cpu().detach().numpy())
        # np.save("holder/output_waveform.npy", self.hifi_gan(output.spectrogram).cpu().detach().numpy())
        # np.save("holder/spectrogram_prenet.npy", output.spectrogram_prenet.cpu().detach().numpy())
        
        self.log(
            "train/loss",
            output.loss,
            on_step=True,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(jp_audio),
        )
        self.log(
            "train/l1_loss",
            output.l1_loss,
            on_step=True,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(jp_audio),
        )
        self.log(
            "train/bce_loss",
            output.bce_loss,
            on_step=True,
            # on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(jp_audio),
        )
        # self.log(
        #     "train/l2_loss",
        #     output.l2_loss,
        #     on_step=True,
        #     # on_epoch=True,
        #     prog_bar=True,
        #     logger=True,
        #     batch_size=len(jp_audio),
        # )
        
        
        if self.global_step % 500 == 0:
            # Log the spectrograms
            self.logger.experiment.add_image(
                "train/spectrogram",
                torch.exp(output.spectrogram[0]),
                global_step=self.global_stepd,
                dataformats="WH",
            )

            # Log the spectrogram prenet
            self.logger.experiment.add_image(
                "train/spectrogram_prenet",
                torch.exp(output.spectrogram_prenet[0]),
                global_step=self.global_step,
                dataformats="WH",
            )

            # Load the output label spectrogram
            self.logger.experiment.add_image(
                "train/labels",
                torch.exp(features[0]),
                global_step=self.global_step,
                dataformats="WH",
            )


            if output.spectrogram is not None:
                # spectrogram = output.spectrogram[0].unsqueeze(0)
                waveform = self.hifi_gan(output.spectrogram)
                
                if waveform.shape[0] > 0:
                    # Log generated waveform
                    self.logger.experiment.add_audio(
                        "train/generated_waveform",
                        waveform[0],
                        global_step=self.global_step,
                        sample_rate=16000,
                    )

                    # Assuming jp_audio or en_audio is the ground truth audio
                    # Adjust this part as per your actual ground truth data structure
                    ground_truth_audio = en_audio[0]

                    # Log ground truth audio
                    self.logger.experiment.add_audio(
                        "train/ground_truth_audio",
                        ground_truth_audio,
                        global_step=self.global_step,
                        sample_rate=16000,  # Adjust this sample rate if necessary
                    )


        return output.loss

    def validation_step(self, batch, batch_idx):
        jp_audio, en_audio = batch

        output = self(jp_audio, en_audio)
        output, features = output

        self.log(
            "val/loss",
            output.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(jp_audio),
        )

        if output.spectrogram is not None and self.global_step % 1000 == 0:
            spectrogram =  output.spectrogram[0].unsqueeze(0)
            waveform = self.hifi_gan(spectrogram)

            if waveform.shape[0] > 0:
                self.logger.experiment.add_audio(
                    "val/waveform",
                    waveform[0],
                    global_step=self.global_step,
                    sample_rate=16000,
                )

        return output.loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=250, gamma=0.9),
            "interval": "epoch",
        }
        return [optimizer], [scheduler]
        # return [optimizer]

    def custom_collate_fn(self, batch):
        jp_audio_tensors = [item[0] for item in batch]
        en_audio_tensors = [item[1] for item in batch]

        return jp_audio_tensors, en_audio_tensors

    def train_dataloader(self):
        train_data = AudioDataset(mode="train")
        return DataLoader(
            train_data,
            batch_size=3,
            shuffle=True,
            collate_fn=self.custom_collate_fn,
            num_workers=18,
        )

    def val_dataloader(self):
        val_data = AudioDataset(mode="val")
        return DataLoader(
            val_data,
            batch_size=16,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
            num_workers=4,
        )

    # def on_epoch_end(self):
    #     # Log learning rate at the end of each epoch
    #     for idx, scheduler in enumerate(self.lr_schedulers()):
    #         lr = scheduler.get_last_lr()[0]
    #         self.log('lr', lr, on_epoch=True, logger=True)


speech_t5_module = SpeechT5Module()

logger = TensorBoardLogger("hifigan_logs", name="l2_loss_1e-4")
# checkpoint_callback = ModelCheckpoint(
#      monitor='val_loss',
#      dirpath='my/path/',
#      filename='sample-mnist-{epoch:02d}-{val_loss:.2f}'
# )

trainer = Trainer(
    max_epochs=15000,
    accelerator="gpu",
    strategy="ddp_find_unused_parameters_true",
    logger=logger,
    # accumulate_grad_batches=5,
    callbacks=[LearningRateMonitor(logging_interval='step')],
    overfit_batches=1,
    check_val_every_n_epoch=int(1e6),
)

trainer.fit(speech_t5_module)
