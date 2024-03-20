import glob
import os
import tqdm


# files = glob.glob('/data/crunchyroll/audio_clips_mono_16k_np/**/*.npy', recursive=True)
# same_pairs = {}
# for file in tqdm.tqdm(files):
    
#     split_path = file.split('/')
#     basename = split_path[-2]
#     file_name = split_path[-1].replace('.npy', '')
#     if basename not in same_pairs:
#         same_pairs[basename] = {}
#     if int(file_name)%2 == 0:
#         same_pairs[basename]["jp"] = file
#     else:
#         same_pairs[basename]["en"] = file
        
# ratio = .01
# training = "training.tsv"
# valid = "valid.tsv"

# import random
# for pair in tqdm.tqdm(same_pairs):
#     if random.random() < ratio:
#         with open(valid, 'a') as f:
#             f.write(f"{same_pairs[pair]['jp']}\t{same_pairs[pair]['en']}\n")
#     else:
#         with open(training, 'a') as f:
#             f.write(f"{same_pairs[pair]['jp']}\t{same_pairs[pair]['en']}\n")


# /data/crunchyroll/audio_clips_mono/8 Man After/8 Man After/1.0/clip_pairs_250088/500176.mp3
# files = glob.glob('/data/crunchyroll/audio_clips_mono/**/*.mp3', recursive=True)
# same_pairs = {}
# for file in tqdm.tqdm(files):
    
#     split_path = file.split('/')
#     basename = split_path[-2]
#     file_name = split_path[-1].replace('.mp3', '')
#     if basename not in same_pairs:
#         same_pairs[basename] = {}
#     if int(file_name)%2 == 0:
#         same_pairs[basename]["jp"] = file
#     else:
#         same_pairs[basename]["en"] = file
        
# ratio = .01
# training = "training_mp3.tsv"
# valid = "valid_mp3.tsv"

# import random
# for pair in tqdm.tqdm(same_pairs):
#     if random.random() < ratio:
#         with open(valid, 'a') as f:
#             f.write(f"{same_pairs[pair]['jp']}\t{same_pairs[pair]['en']}\n")
#     else:
#         with open(training, 'a') as f:
#             f.write(f"{same_pairs[pair]['jp']}\t{same_pairs[pair]['en']}\n")


training = "training_mp3.tsv"
valid = "valid_mp3.tsv"

with open("training.tsv", 'r') as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        jp, en = line.strip().split('\t')
        jp = jp.replace('audio_clips_mono_16k_np', 'audio_clips_mono')
        en = en.replace('audio_clips_mono_16k_np', 'audio_clips_mono')
        jp = jp.replace('.npy', '.mp3')
        en = en.replace('.npy', '.mp3')
        with open(training, 'a') as f:
            f.write(f"{jp}\t{en}\n")

with open("valid.tsv", 'r') as f:
    lines = f.readlines()
    for line in tqdm.tqdm(lines):
        jp, en = line.strip().split('\t')
        jp = jp.replace('audio_clips_mono_16k_np', 'audio_clips_mono')
        en = en.replace('audio_clips_mono_16k_np', 'audio_clips_mono')
        jp = jp.replace('.npy', '.mp3')
        en = en.replace('.npy', '.mp3')
        with open(valid, 'a') as f:
            f.write(f"{jp}\t{en}\n")