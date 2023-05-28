#!/usr/bin/env python3
from transformers import OPTForCausalLM, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-350m')
model = OPTForCausalLM.from_pretrained('facebook/opt-350m')

# begin tokens
start_tokens = torch.tensor(2 * [[[0]]])
print(start_tokens.shape[-1])

for i in range(start_tokens.shape[-1]):
    out_tokens = model.generate(start_tokens[i])
    opt_embeddings = model.get_input_embeddings()
    # generated_embedding_vectors has shape [len(opt_embeddings), hidden_size]
    generated_embedding_vectors = opt_embeddings(out_tokens)[0]