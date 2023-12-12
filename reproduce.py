# Reproduce https://github.com/huggingface/transformers/issues/26598

from transformers import SpeechT5Processor, SpeechT5ForSpeechToSpeech
import numpy as np

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_vc")
model = SpeechT5ForSpeechToSpeech.from_pretrained("microsoft/speecht5_vc")

features = processor(audio=[np.random.random(size=(2048,)) for waveform in range(3)], audio_target=[np.random.random(size=(2048,)) for waveform in range(3)], return_tensors="pt", padding=True, sampling_rate=16000)
outputs = model(**features, return_dict=True)


# Traceback (most recent call last):
#   File "[REDACTED]/reproduce.py", line 8, in <module>
#     outputs = model(**features, return_dict=True)
#   File "/[REDACTED]/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/[REDACTED]/torch/nn/modules/module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/[REDACTED]/transformers/models/speecht5/modeling_speecht5.py", line 2953, in forward
#     outputs = self.speecht5(
#   File "/[REDACTED]/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/[REDACTED]/torch/nn/modules/module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/[REDACTED]/transformers/models/speecht5/modeling_speecht5.py", line 2211, in forward
#     decoder_outputs = self.decoder(
#   File "/[REDACTED]/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/[REDACTED]/torch/nn/modules/module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/[REDACTED]/transformers/models/speecht5/modeling_speecht5.py", line 1734, in forward
#     outputs = self.wrapped_decoder(
#   File "/[REDACTED]/torch/nn/modules/module.py", line 1518, in _wrapped_call_impl
#     return self._call_impl(*args, **kwargs)
#   File "/[REDACTED]/torch/nn/modules/module.py", line 1527, in _call_impl
#     return forward_call(*args, **kwargs)
#   File "/[REDACTED]/transformers/models/speecht5/modeling_speecht5.py", line 1594, in forward
#     attention_mask = _prepare_4d_causal_attention_mask(
#   File "/[REDACTED]/transformers/modeling_attn_mask_utils.py", line 195, in _prepare_4d_causal_attention_mask
#     attention_mask = attn_mask_converter.to_4d(
#   File "/[REDACTED]/transformers/modeling_attn_mask_utils.py", line 117, in to_4d
#     expanded_4d_mask = expanded_attn_mask if causal_4d_mask is None else expanded_attn_mask + causal_4d_mask
# RuntimeError: The size of tensor a (9) must match the size of tensor b (4) at non-singleton dimension 3
