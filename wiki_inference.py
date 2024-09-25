import torch
from llama.model import Transformer, ModelArgs
from llama.tokenizer import Tokenizer
import math
import json
from datasets import load_dataset
from tqdm import tqdm
import argparse

def load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, hopformer, win_size, sim_thresh):
    # Load the model arguments and state dict
    with open(f"{ckpt_dir}/params.json", "r") as f:
        params = json.loads(f.read())
    
    model_args = ModelArgs(
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
        hopformer=hopformer,
        win_size=win_size,
        sim_thresh=sim_thresh,
        **params
    )
    
    model = Transformer(model_args)
    
    # Load the checkpoint
    checkpoint = torch.load(f"{ckpt_dir}/consolidated.00.pth", map_location="cpu")
    model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    tokenizer = Tokenizer(model_path=tokenizer_path)
    
    return model, tokenizer

def calculate_perplexity(model, tokenizer, text, max_seq_len):
    tokens = tokenizer.encode(text, bos=True, eos=False)
    
    if len(tokens) > max_seq_len:
        tokens = tokens[:max_seq_len]
    
    tokens = torch.tensor(tokens).unsqueeze(0)
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    
    with torch.no_grad():
        logits = model(tokens, start_pos=0)
    
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0, :-1].gather(1, tokens[0, 1:].unsqueeze(1)).squeeze(1)
    
    return -token_log_probs.sum().item(), len(tokens) - 1

def evaluate_wikitext(model, tokenizer, max_seq_len):
    # Load WikiText-2 test dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    total_log_likelihood = 0
    total_tokens = 0
    
    for example in tqdm(dataset, desc="Evaluating"):
        text = example["text"]
        if len(text.strip()) == 0:  # Skip empty lines
            continue
        
        log_likelihood, num_tokens = calculate_perplexity(model, tokenizer, text, max_seq_len)
        total_log_likelihood += log_likelihood
        total_tokens += num_tokens
    
    perplexity = math.exp(total_log_likelihood / total_tokens)
    return perplexity

def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama2 model on WikiText-2 dataset")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Maximum batch size")
    parser.add_argument("--hopformer", action="store_true", help="Use HopFormer attention")
    parser.add_argument("--win_size", type=int, default=30, help="Window size for HopFormer")
    parser.add_argument("--sim_thresh", type=float, default=0.5, help="Similarity threshold for HopFormer")
    
    args = parser.parse_args()
    
    model, tokenizer = load_model(
        args.ckpt_dir,
        args.tokenizer_path,
        args.max_seq_len,
        args.max_batch_size,
        args.hopformer,
        args.win_size,
        args.sim_thresh
    )
    
    perplexity = evaluate_wikitext(model, tokenizer, args.max_seq_len)
    print(f"WikiText-2 Perplexity: {perplexity:.2f}")
    print(f"HopFormer: {'Enabled' if args.hopformer else 'Disabled'}")
    if args.hopformer:
        print(f"Window Size: {args.win_size}")
        print(f"Similarity Threshold: {args.sim_thresh}")

if __name__ == "__main__":
    main()