"""
Sample from a trained model
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
init_from = 'resume'  # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')

# IMPORTANT: you are running from .../nanoGPT/nanoGPT but your checkpoint is in .../nanoGPT/out
# so point to the parent folder's out directory:
out_dir = '../nanoGPT/out'    # <-- changed from 'out'

start = "\n"          # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10      # number of samples to draw
max_new_tokens = 500  # number of tokens generated in each sample
temperature = 0.8     # 1.0 = no change, < 1.0 = less random, > 1.0 = more random
top_k = 200           # retain only the top_k most likely tokens
seed = 1337

# Mac-friendly device selection:
# - Apple Silicon: use "mps" (Metal)
# - Otherwise: CPU
device = 'mps' if torch.backends.mps.is_available() else 'cpu'

# For maximum compatibility on Mac, use float32 (avoids autocast quirks on some torch builds)
dtype = 'float32'

compile = False  # torch.compile can be flaky on some Mac setups; keep off by default
exec(open('configurator.py').read())  # overrides from command line or config file
# -----------------------------------------------------------------------------

torch.manual_seed(seed)

device_type = 'mps' if device == 'mps' else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

# Safest: disable autocast on Mac unless you explicitly want it
ctx = nullcontext()

# model
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            f"Try setting out_dir to the folder that contains ckpt.pt."
        )

    # Robust: always load checkpoint to CPU first to avoid CUDA deserialization issues
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']

    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)

elif init_from.startswith('gpt2'):
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)

if compile:
    model = torch.compile(model)

# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)

if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()

start_ids = encode(start)
x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
