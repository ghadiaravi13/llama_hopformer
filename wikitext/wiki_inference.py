import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# import pdb; pdb.set_trace()

import torch
from llama.model import Transformer, ModelArgs
from llama.tokenizer import Tokenizer
import math
import json
from datasets import load_dataset
from tqdm import tqdm
import argparse
from safetensors import safe_open
from transformers import AutoTokenizer

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)

def load_model(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, hopformer, win_size, sim_thresh):
    # Load the model configuration
    with open(os.path.join(ckpt_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    model_args = ModelArgs(
        max_batch_size=max_batch_size,
        hopformer=hopformer,
        win_size=win_size,
        sim_thresh=sim_thresh,
        dim=config["hidden_size"],
        n_layers=config["num_hidden_layers"],
        n_heads=config["num_attention_heads"],
        vocab_size=config["vocab_size"],
        multiple_of=config.get("multiple_of", 256),
        norm_eps=config.get("rms_norm_eps", 1e-5),
        max_seq_len=config["max_position_embeddings"],
    )
    
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group("nccl", world_size=1, rank=0)
    if not model_parallel_is_initialized():
        # if model_parallel_size is None:
        model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
        initialize_model_parallel(model_parallel_size)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    model = Transformer(model_args)
    
    # Load the checkpoint
    tensors = {}
    for i in range(1, 3):  # Load both parts of the model
        with safe_open(os.path.join(ckpt_dir, f"model-0000{i}-of-00002.safetensors"), framework="pt", device="cpu") as f:
            for k in f.keys():
                tensors[k] = f.get_tensor(k)
    
    model.load_state_dict(tensors, strict=False)
    
    model.eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir)
    
    return model, tokenizer

def calculate_perplexity(model, tokenizer, text, max_seq_len):
    # Use the tokenizer's encode_plus method
    encoded = tokenizer.encode_plus(text, add_special_tokens=True, truncation=True, max_length=max_seq_len, return_tensors="pt")
    tokens = encoded['input_ids']
    
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    
    with torch.no_grad():
        logits = model(tokens, start_pos=0)
    
    log_probs = torch.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[0, :-1].gather(1, tokens[0, 1:].unsqueeze(1)).squeeze(1)
    
    return -token_log_probs.sum().item(), len(tokens[0]) - 1

def evaluate_wikitext(model, tokenizer, max_seq_len):
    # Load WikiText-2 test dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    total_log_likelihood = 0
    total_tokens = 0
    results = []

    counter = 0
    
    for example in tqdm(dataset, desc="Evaluating"):
        
        if counter > 10:
            break
        counter += 1

        text = example["text"]
        if len(text.strip()) == 0:  # Skip empty lines
            continue
        
        log_likelihood, num_tokens = calculate_perplexity(model, tokenizer, text, max_seq_len)
        total_log_likelihood += log_likelihood
        total_tokens += num_tokens
        
        results.append({
            "text": text,
            "log_likelihood": log_likelihood,
            "num_tokens": num_tokens,
            "perplexity": math.exp(log_likelihood / num_tokens)
        })
    
    perplexity = math.exp(total_log_likelihood / total_tokens)
    return perplexity, results

def main():
    parser = argparse.ArgumentParser(description="Evaluate Llama2 model on WikiText-2 dataset")
    parser.add_argument("--ckpt_dir", type=str, required=True, help="Path to the checkpoint directory")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer model")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--max_batch_size", type=int, default=1, help="Maximum batch size")
    parser.add_argument("--hopformer", action="store_true", help="Use HopFormer attention")
    parser.add_argument("--win_size", type=int, default=10, help="Window size for HopFormer")
    parser.add_argument("--sim_thresh", type=float, default=0.5, help="Similarity threshold for HopFormer")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Redirect print statements to log file
    log_file = os.path.join(args.output_dir, "wikitext_ppl.log")
    sys.stdout = open(log_file, "w")
    
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
        print(f"Running in distributed mode with world_size: {world_size}, local_rank: {local_rank}")
    else:
        print("Running in non-distributed mode")
    
    model, tokenizer = load_model(
        args.ckpt_dir,
        args.tokenizer_path,
        args.max_seq_len,
        args.max_batch_size,
        args.hopformer,
        args.win_size,
        args.sim_thresh
    )
    
    # Only evaluate on rank 0 in distributed mode
    if world_size == 1 or local_rank == 0:
        perplexity, results = evaluate_wikitext(model, tokenizer, args.max_seq_len)
        print(f"WikiText-2 Perplexity: {perplexity:.2f}")
        print(f"HopFormer: {'Enabled' if args.hopformer else 'Disabled'}")
        if args.hopformer:
            print(f"Window Size: {args.win_size}")
            print(f"Similarity Threshold: {args.sim_thresh}")
        
        # Save results to JSON file
        json_file = os.path.join(args.output_dir, "wikitext_ppl.json")
        with open(json_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {json_file}")
        print(f"Log saved to {log_file}")

if __name__ == "__main__":
    main()