import warnings
from typing import Optional, Tuple, Union
from pytorch_lightning import LightningModule, Trainer
from transformers import SpeechT5Processor
from transformers.models.speecht5.modeling_speecht5 import SpeechT5SpectrogramLoss, SpeechT5SpeechDecoderPostnet, SpeechT5PreTrainedModel, SpeechT5Config, SpeechT5DecoderWithSpeechPrenet, SpeechT5EncoderWithSpeechPrenet, SpeechT5Model, Seq2SeqSpectrogramOutput, shift_spectrograms_right, _generate_speech

from datasets.load_datasets import AudioDataset

import torch
from torch.utils.data import Dataset as TorchDataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class SpeechT5ForSpeechToSpeech(SpeechT5PreTrainedModel):
    def __init__(self, config: SpeechT5Config):
        super().__init__(config)
        config.use_guided_attention_loss = False

        speech_encoder = SpeechT5EncoderWithSpeechPrenet(config)
        speech_decoder = SpeechT5DecoderWithSpeechPrenet(config)
        self.speecht5 = SpeechT5Model(config, speech_encoder, speech_decoder)
        self.speech_decoder_postnet = SpeechT5SpeechDecoderPostnet(config)
        self.criterion = SpeechT5SpectrogramLoss(config)

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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if stop_labels is not None:
            warnings.warn(
                "The argument `stop_labels` is deprecated and will be removed in version 4.30.0 of Transformers",
                FutureWarning,
            )

        if labels is not None:
            if decoder_input_values is None:
                decoder_input_values = shift_spectrograms_right(labels, self.config.reduction_factor)

                # Downsample the decoder attention mask, as the decoder reduction factor is not always 1, and shift the mask to the right
                if decoder_attention_mask is not None:
                    if self.config.reduction_factor > 1:
                        decoder_attention_mask = decoder_attention_mask[:, self.config.reduction_factor - 1 :: self.config.reduction_factor]


                    shifted_attention_mask = decoder_attention_mask.new_zeros(decoder_attention_mask.shape)
                    shifted_attention_mask[:, 1:] = decoder_attention_mask[:, :-1].clone()
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
            output_attentions=output_attentions or self.config.use_guided_attention_loss,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        spectrogram_prenet, spectrogram, logits = self.speech_decoder_postnet(outputs[0])

        loss = None
        if labels is not None:

            # Ensure that the outputs and labels have the same sequence length -- This could be different since the
            # reduction factor is not always 1
            labels = labels[:, :-1]

            if labels.shape[1] != spectrogram_prenet.shape[1]:
                if labels.shape[1] > spectrogram_prenet.shape[1]:
                    labels = labels[:, :spectrogram_prenet.shape[1]]
                else:
                    spectrogram_prenet = spectrogram_prenet[:, :labels.shape[1]]
                    spectrogram = spectrogram[:, :labels.shape[1]]
                    logits = logits[:, :labels.shape[1]]

            loss = self.criterion(
                attention_mask,
                spectrogram_prenet,
                spectrogram,
                logits,
                labels,
                outputs.cross_attentions,
            )

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

        self.processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
        self.model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")

    def forward(self, jp_audio, en_audio):

        features = self.processor(audio=[
            waveform.squeeze().cpu().numpy() for waveform in jp_audio
        ], audio_target=[
            waveform.squeeze().cpu().numpy() for waveform in en_audio
        ], return_tensors="pt", padding=True, sampling_rate=16000)
        assert features is not None

        for key, value in features.items():
            features[key] = value.to(self.device)

        outputs = self.model(**features)

        return outputs.loss

    def training_step(self, batch, batch_idx):

        jp_audio, en_audio = batch
        output = self(jp_audio, en_audio)

        self.log('train/loss', output, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(jp_audio))

        return output

    def validation_step(self, batch, batch_idx):
        jp_audio, en_audio = batch

        output = self(jp_audio, en_audio)

        self.log('val/loss', output, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=len(jp_audio))

        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


    def custom_collate_fn(self, batch):
        jp_audio_tensors = [item[0] for item in batch]
        en_audio_tensors = [item[1] for item in batch]

        return jp_audio_tensors, en_audio_tensors

    def train_dataloader(self):
        train_data = AudioDataset(mode='train')
        return DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=self.custom_collate_fn, num_workers=18)

    def val_dataloader(self):
        val_data = AudioDataset(mode='val')
        return DataLoader(val_data, batch_size=32, shuffle=False, collate_fn=self.custom_collate_fn, num_workers=4)


speech_t5_module = SpeechT5Module()

trainer = Trainer(max_epochs=10,
                  accelerator='gpu',
                  strategy='ddp_find_unused_parameters_true')

trainer.fit(speech_t5_module)
