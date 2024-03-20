import warnings
from typing import Optional, Tuple, Union
from lightning.pytorch import LightningModule, Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.utilities.types import EVAL_DATALOADERS

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

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np

import os

import auraloss

from anim_datasets.load_datasets import AudioDataset

class SpeechT5ForSpeechToSpeech(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config, sample_rate: int = 16000):
        super().__init__(config)
        config.use_guided_attention_loss = False

        speech_encoder = SpeechT5EncoderWithSpeechPrenet(config)
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        self.speecht5 = SpeechT5Model(config, speech_encoder, speech_decoder)
        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)

        self.criterion = SpeechT5SpectrogramLoss(config)

        # self.loss_fn = auraloss.freq.MultiResolutionSTFTLoss(
        #     fft_sizes=[1024, 2048, 8192],
        #     hop_sizes=[256, 512, 2048],
        #     win_lengths=[1024, 2048, 8192],
        #     scale="mel",
        #     n_bins=128,
        #     sample_rate=sample_rate,
        #     perceptual_weighting=True,
        # )
        self.loss_fn =  auraloss.freq.MultiResolutionSTFTLoss()

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.speecht5.get_encoder()

    def get_decoder(self):
        return self.speecht5.get_decoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.get_encoder().prenet.freeze_feature_encoder()

    def forward(
        self,
        input_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_values: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        stop_labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, Seq2SeqSpectrogramOutput]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if stop_labels is not None:
            warnings.warn(
                "The argument `stop_labels` is deprecated and will be removed in version 4.30.0 of Transformers",
                FutureWarning,
            )

        if labels is not None:
            if decoder_input_values is None:
                decoder_input_values = shift_spectrograms_right(
                    labels, self.config.reduction_factor
                )

                # Downsample the decoder attention mask, as the decoder reduction factor is not always 1, and shift the mask to the right
                if decoder_attention_mask is not None:
                    if self.config.reduction_factor > 1:
                        decoder_attention_mask = decoder_attention_mask[
                            :,
                            self.config.reduction_factor
                            - 1 :: self.config.reduction_factor,
                        ]

                    shifted_attention_mask = decoder_attention_mask.new_zeros(
                        decoder_attention_mask.shape
                    )
                    shifted_attention_mask[:, 1:] = decoder_attention_mask[
                        :, :-1
                    ].clone()
                    decoder_attention_mask = shifted_attention_mask

        outputs = self.speecht5(
            input_values=input_values,
            attention_mask=attention_mask,
            decoder_input_values=decoder_input_values,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            encoder_outputs=encoder_outputs,
            past_key_values=past_key_values,
            use_cache=use_cache,
            speaker_embeddings=speaker_embeddings,
            output_attentions=output_attentions
            or self.config.use_guided_attention_loss,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        spectrogram_prenet, spectrogram, logits = self.speech_decoder_postnet(
            outputs[0]
        )

        loss = None
        if labels is not None:
            # Ensure that the outputs and labels have the same sequence length -- This could be different since the
            # reduction factor is not always 1
            labels = labels[:, :-1]

            if labels.shape[1] != spectrogram_prenet.shape[1]:
                if labels.shape[1] > spectrogram_prenet.shape[1]:
                    labels = labels[:, : spectrogram_prenet.shape[1]]
                else:
                    spectrogram_prenet = spectrogram_prenet[:, : labels.shape[1]]
                    spectrogram = spectrogram[:, : labels.shape[1]]
                    logits = logits[:, : labels.shape[1]]
            
            loss, l1_loss, bce_loss = self.criterion(
                attention_mask,
                spectrogram_prenet,
                spectrogram,
                logits,
                labels,
                outputs.cross_attentions,
            )
            # spectrogram
            # loss = self.loss_fn(spectrogram.T, labels.T)

        if not return_dict:
            output = (spectrogram,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        
        return Seq2SeqSpectrogramOutput(
            loss=loss,
            spectrogram=spectrogram,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
            l1_loss=l1_loss,
            bce_loss=bce_loss,
            spectrogram_prenet=spectrogram_prenet,
            
        )

    @torch.no_grad()
    def generate_speech(
        self,
        input_values: torch.FloatTensor,
        speaker_embeddings: Optional[torch.FloatTensor] = None,
        threshold: float = 0.5,
        minlenratio: float = 0.0,
        maxlenratio: float = 20.0,
        vocoder: Optional[torch.nn.Module] = None,
        output_cross_attentions: bool = False,
    ) -> torch.FloatTensor:
        if speaker_embeddings is None:
            speaker_embeddings = torch.zeros((1, 512), device=input_values.device)

        return _generate_speech(
            self,
            input_values,
            speaker_embeddings,
            threshold,
            minlenratio,
            maxlenratio,
            vocoder,
            output_cross_attentions,
        )


class SpeechT5Module(LightningModule):
    def __init__(self):
        super().__init__()
        self.save_hyperparameters()        
        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        self.processor.do_normalize = True
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")
        self.hifi_gan = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").eval()
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
        self.last_train_output = output, jp_audio, en_audio

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

        
        
        if self.global_step % 1e7 == 0:
            # Log the spectrograms
            self.logger.experiment.add_image(
                "train/spectrogram",
                torch.exp(output.spectrogram[0]),
                global_step=self.global_step,
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


            # if output.spectrogram is not None:
            #     # spectrogram = output.spectrogram[0].unsqueeze(0)
            #     waveform = self.hifi_gan(output.spectrogram)
                
            #     if waveform.shape[0] > 0:
            #         # Log generated waveform
            #         self.logger.experiment.add_audio(
            #             "train/generated_waveform",
            #             waveform[0],
            #             global_step=self.global_step,
            #             sample_rate=16000,
            #         )

            #         # Assuming jp_audio or en_audio is the ground truth audio
            #         # Adjust this part as per your actual ground truth data structure
            #         ground_truth_audio = en_audio[0]

            #         # Log ground truth audio
            #         self.logger.experiment.add_audio(
            #             "train/ground_truth_audio",
            #             ground_truth_audio,
            #             global_step=self.global_step,
            #             sample_rate=16000,  # Adjust this sample rate if necessary
            #         )


        return output.loss

    def validation_step(self, batch, batch_idx):
        jp_audio, en_audio = batch

        output = self(jp_audio, en_audio)
        
        self.last_val_output = output, jp_audio, en_audio

        output, features = output
        features = features["labels"]

        self.log(
            "val/loss",
            output.loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(jp_audio),
            sync_dist=True
        )
        self.log(
            "val/l1_loss",
            output.l1_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(jp_audio),
            sync_dist=True
        )
        self.log(
            "val/bce_loss",
            output.bce_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=len(jp_audio),
            sync_dist=True
        )

        
        
        # if self.global_step % 1e7 == 0:
        #     # Log the spectrograms
        #     self.logger.experiment.add_image(
        #         "val/spectrogram",
        #         torch.exp(output.spectrogram[0]),
        #         global_step=self.global_step,
        #         dataformats="WH",
        #     )

        #     # Log the spectrogram prenet
        #     self.logger.experiment.add_image(
        #         "val/spectrogram_prenet",
        #         torch.exp(output.spectrogram_prenet[0]),
        #         global_step=self.global_step,
        #         dataformats="WH",
        #     )

        #     # Load the output label spectrogram
        #     self.logger.experiment.add_image(
        #         "val/labels",
        #         torch.exp(features[0]),
        #         global_step=self.global_step,
        #         dataformats="WH",
        #     )


            # if output.spectrogram is not None:
            #     # spectrogram = output.spectrogram[0].unsqueeze(0)
            #     waveform = self.hifi_gan(output.spectrogram)
                
            #     if waveform.shape[0] > 0:
            #         # Log generated waveform
            #         self.logger.experiment.add_audio(
            #             "val/generated_waveform",
            #             waveform[0],
            #             global_step=self.global_step,
            #             sample_rate=16000,
            #         )

            #         # Assuming jp_audio or en_audio is the ground truth audio
            #         # Adjust this part as per your actual ground truth data structure
            #         ground_truth_audio = en_audio[0]

            #         # Log ground truth audio
            #         self.logger.experiment.add_audio(
            #             "val/ground_truth_audio",
            #             ground_truth_audio,
            #             global_step=self.global_step,
            #             sample_rate=16000,  # Adjust this sample rate if necessary
            #         )


        return output.loss

    def configure_optimizers(self):

        # Define the number of warm-up steps
        warmup_steps = 4000
        # Maximum learning rate after warm-up
        max_lr = 5e-5
        gamma = .99999
        # Assume `total_steps` is accessible and tracks the global step count across epochs

        optimizer = torch.optim.Adam(self.parameters(), lr=max_lr)

        
        # Lambda function to adjust the learning rate based on global steps
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            # Apply your desired schedule post warm-up here
            # Example: Continue with max_lr without decay
            else:
                return gamma ** (current_step - warmup_steps)

        # Update the scheduler to use the step-based lambda function
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda current_step: lr_lambda(current_step))

        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
        # return [optimizer]

    def custom_collate_fn(self, batch):
        jp_audio_tensors = [item[0] for item in batch]
        en_audio_tensors = [item[1] for item in batch]

        return jp_audio_tensors, en_audio_tensors

    def train_dataloader(self):
        train_data = AudioDataset(mode="train")
        return DataLoader(
            train_data,
            batch_size=20,
            shuffle=True,
            collate_fn=self.custom_collate_fn,
            num_workers=18,
        )

    def val_dataloader(self):
        val_data = AudioDataset(mode="val")
        return DataLoader(
            val_data,
            batch_size=8,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
            num_workers=4,
        )
        
    def test_dataloader(self):
        test_data = AudioDataset(mode="test")
        return DataLoader(
            test_data,
            batch_size=8,
            shuffle=False,
            collate_fn=self.custom_collate_fn,
            num_workers=4,
        )
    
    def log_audio_and_spectrogram(self, spectrogram, waveform, ground_truth_audio, labels, phase):
        # Helper function to log audio and spectrogram
        if spectrogram is not None and waveform.shape[0] > 0:
            # Log generated waveform
            self.logger.experiment.add_audio(
                f"{phase}/generated_waveform",
                waveform[0],
                global_step=self.current_epoch,
                sample_rate=16000,
            )

            # Log ground truth audio
            self.logger.experiment.add_audio(
                f"{phase}/ground_truth_audio",
                ground_truth_audio,
                global_step=self.current_epoch,
                sample_rate=16000,
            )

            # Log spectrogram as an image, assuming spectrogram is a tensor you wish to visualize
            # Adjust tensor operations as needed for your visualization purposes
            spectrogram_image = torch.exp(spectrogram[0])  # Example transformation, adjust as needed
            self.logger.experiment.add_image(
                f"{phase}/spectrogram",
                spectrogram_image,
                self.current_epoch,
                dataformats="WH",
            )

            # Log the output label spectrogram
            self.logger.experiment.add_image(
                f"{phase}/labels",
                torch.exp(labels[0]),
                self.current_epoch,
                dataformats="WH",
            )

    def on_train_epoch_end(self, unused=None):
        # Assuming you have stored the last training batch's output in self.last_train_output
        if hasattr(self, 'last_train_output'):
            
            output, jp_audio, en_audio = self.last_train_output
            output, feat = output
            features = feat["labels"]
            spectrogram = output.spectrogram if 'spectrogram' in output else None
            waveform = self.hifi_gan(spectrogram) if spectrogram is not None else None
            ground_truth_audio = en_audio[0]  # Adjust this part as per your actual data structure

            self.log_audio_and_spectrogram(spectrogram, waveform, ground_truth_audio, features, "train")

    def on_validation_epoch_end(self, unused=None):
        # Assuming you have stored the last validation batch's output in self.last_val_output
        if hasattr(self, 'last_val_output'):
            output, jp_audio, en_audio = self.last_val_output
            output, feat = output
            features = feat["labels"]
            spectrogram = output.spectrogram if 'spectrogram' in output else None
            waveform = self.hifi_gan(spectrogram) if spectrogram is not None else None
            ground_truth_audio = en_audio[0]  # Adjust this part as per your actual data structure

            self.log_audio_and_spectrogram(spectrogram, waveform, ground_truth_audio, features, "val")
        

if __name__ == "__main__":
    speech_t5_module = SpeechT5Module()

    name = "logs/warmup_scheduler"
    ver = 0
    dir = f"/data/kevincai/ckpt/{name}/version_{ver}"
    while os.path.exists(dir):
        ver += 1
        dir = f"/data/kevincai/ckpt/{name}/version_{ver}"


    logger = TensorBoardLogger(name, name="")
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"/data/kevincai/ckpt/{name}/version_{ver}",
        monitor='val/loss',
        filename='sample-{epoch:02d}-{val_loss:.2f}',
        every_n_epochs=1,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=15000,
        accelerator="gpu",
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        accumulate_grad_batches=30,
        callbacks=[LearningRateMonitor(logging_interval='step'), checkpoint_callback],
        # check_val_every_n_epoch=1,
        enable_checkpointing=True,
    )

    trainer.fit(speech_t5_module)
