import json
import os
import sys
import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_benchmark.datasets.loader import load_dataset_by_name
PROMPT_TPL = 'Please solve the following problem step by step. Show your reasoning and put your final answer in \\boxed{{}}.\n\nProblem: {question}'

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    ap.add_argument('--dataset', default='math500')
    ap.add_argument('--out', default='results/prefill_hidden_math500.npz')
    ap.add_argument('--n_pool', type=int, default=5, help='Number of last-prompt-tokens to average for feature.')
    args = ap.parse_args()
    print(f'Loading {args.dataset}...')
    problems = load_dataset_by_name(args.dataset)
    print(f'  {len(problems)} problems')
    print(f'Loading tokenizer + model from {args.model}...')
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=False)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True, local_files_only=False)
    model.eval()
    print(f'  Hidden size: {model.config.hidden_size}')
    features = []
    pids = []
    questions = []
    answers_gt = []
    prompt_lens = []
    with torch.no_grad():
        for i, p in enumerate(problems):
            prompt = PROMPT_TPL.format(question=p.question)
            inputs = tok(prompt, return_tensors='pt', truncation=True, max_length=2048).to('cuda')
            outputs = model(**inputs, output_hidden_states=True, use_cache=False)
            last_hidden = outputs.hidden_states[-1][0]
            seq_len = last_hidden.shape[0]
            n_pool = min(args.n_pool, seq_len)
            feat = last_hidden[-n_pool:].mean(dim=0)
            features.append(feat.float().cpu().numpy())
            pids.append(p.id)
            questions.append(p.question[:200])
            answers_gt.append(p.answer)
            prompt_lens.append(int(inputs['input_ids'].shape[1]))
            if (i + 1) % 50 == 0:
                print(f'  [{i + 1}/{len(problems)}]')
    features = np.stack(features)
    print(f'Features shape: {features.shape}')
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    np.savez(args.out, features=features, pids=np.array(pids), prompt_lens=np.array(prompt_lens), gt=np.array(answers_gt, dtype=object))
    print(f'Saved to {args.out}')
if __name__ == '__main__':
    main()
