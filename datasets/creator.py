import glob
import os
import tqdm


files = glob.glob('/data/crunchyroll/audio_clips_mono_16k_np/**/*.npy', recursive=True)
same_pairs = {}
for file in tqdm.tqdm(files):
    
    split_path = file.split('/')
    basename = split_path[-2]
    file_name = split_path[-1].replace('.npy', '')
    if basename not in same_pairs:
        same_pairs[basename] = {}
    if int(file_name)%2 == 0:
        same_pairs[basename]["jp"] = file
    else:
        same_pairs[basename]["en"] = file
        
ratio = .01
training = "training.tsv"
valid = "valid.tsv"

import random
for pair in tqdm.tqdm(same_pairs):
    if random.random() < ratio:
        with open(valid, 'a') as f:
            f.write(f"{same_pairs[pair]['jp']}\t{same_pairs[pair]['en']}\n")
    else:
        with open(training, 'a') as f:
            f.write(f"{same_pairs[pair]['jp']}\t{same_pairs[pair]['en']}\n")
