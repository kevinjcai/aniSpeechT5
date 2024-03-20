from diffusers.models.unets.unet_2d import UNet2DModel

from datasets.load_datasets import AudioDataset
from transformers import SpeechT5Processor
from torch.utils.data import Dataset as TorchDataset, DataLoader
import torch

from diffusers import DiffusionPipeline

from diffusers import DiffusionPipeline
import torch



model = UNet2DModel(
    in_channels=1,
    out_channels=1,
).to("cuda")

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
def custom_collate_fn(batch):
    jp_audio_tensors = [item[0] for item in batch]
    en_audio_tensors = [item[1] for item in batch]

    return jp_audio_tensors, en_audio_tensors
dataset = AudioDataset(mode="train")
a = DataLoader(
            dataset,
            batch_size=3,
            shuffle=True,
            num_workers=18,
            collate_fn=custom_collate_fn,
        )

for i in a:
    batch = i
    break
jp_audio, en_audio = batch

features = processor(
            audio=[waveform.squeeze().cpu().numpy() for waveform in jp_audio],
            audio_target=[waveform.squeeze().cpu().numpy() for waveform in en_audio],
            return_tensors="pt",
            padding=True,
            sampling_rate=16000,
            both_spec=True
        )


print(features.keys())
input_vals = features["input_values"].unsqueeze(1)
print(input_vals.shape)
input_vals = input_vals.to("cuda")
tensor_shape = input_vals.size()
n = tensor_shape[2]
m = 8 ** (n - 1).bit_length() 


pad_size = max(0, m - n)
pad_left = pad_size // 2
pad_right = pad_size - pad_left
input_vals = torch.nn.functional.pad(input_vals, (0, 0, pad_left, pad_right))
# input_vals = input_vals.view(3, 1, m, 80)

labels = features["labels"].unsqueeze(1)

print(input_vals.shape)
# print(labels.shape)
output = model(input_vals, timestep=0).sample

print(output.shape)

