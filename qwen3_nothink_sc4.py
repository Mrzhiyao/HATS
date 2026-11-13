from __future__ import annotations
import argparse
import json
import os
import re
import time
from collections import Counter
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from tts_benchmark.datasets.loader import check_answer, load_dataset_by_name, normalize_answer
USER_PROMPT = 'Please solve the following problem step by step. Show your reasoning and put your final answer in \\boxed{{}}.\n\nProblem: {question}'

def format_prompt(tokenizer, question: str) -> str:
    content = USER_PROMPT.format(question=question)
    messages = [{'role': 'user', 'content': content}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template([{'role': 'user', 'content': content + '\n/no_think'}], tokenize=False, add_generation_prompt=True)

def extract_final_answer(text: str) -> str | None:
    text_clean = re.sub('<think>.*?</think>', '', text, flags=re.DOTALL)
    matches = re.findall('\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}', text_clean)
    if matches:
        return matches[-1].strip()
    matches = re.findall('\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}', text)
    return matches[-1].strip() if matches else None

def solve(llm, tokenizer, question: str, gt: str, max_tokens: int):
    prompt = format_prompt(tokenizer, question)
    samples = []
    total_tokens = 0
    params = [SamplingParams(temperature=0.0, max_tokens=max_tokens), SamplingParams(temperature=0.7, top_p=0.95, max_tokens=max_tokens, n=3)]
    t0 = time.time()
    outputs = []
    outputs.extend(llm.generate([prompt], params[0], use_tqdm=False)[0].outputs)
    outputs.extend(llm.generate([prompt], params[1], use_tqdm=False)[0].outputs)
    for out in outputs:
        answer = extract_final_answer(out.text)
        tokens = len(out.token_ids)
        samples.append({'answer': answer, 'tokens': tokens, 'hit_limit': tokens >= max_tokens - 8, 'has_think_tag': '<think>' in out.text})
        total_tokens += tokens
    normalized = [normalize_answer(s['answer']) for s in samples if s['answer']]
    if normalized:
        top, _ = Counter([x for x in normalized if x]).most_common(1)[0]
        final = next((s['answer'] for s in samples if normalize_answer(s['answer']) == top))
    else:
        final = samples[0]['answer']
    return {'n_samples': 4, 'final_answer': final, 'is_correct': check_answer(final or '', gt), 'total_tokens': total_tokens, 'time_ms': int((time.time() - t0) * 1000), 'samples': samples}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True)
    ap.add_argument('--dataset', default='math500')
    ap.add_argument('--n_problems', type=int, default=200)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_tokens', type=int, default=4096)
    ap.add_argument('--max_model_len', type=int, default=5120)
    ap.add_argument('--save_every', type=int, default=5)
    args = ap.parse_args()
    problems = load_dataset_by_name(args.dataset)[:args.n_problems]
    done = {}
    if os.path.exists(args.out):
        done = {r['problem_id']: r for r in json.load(open(args.out))}
        print(f'[resume] {len(done)} done')
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len, gpu_memory_utilization=0.9, trust_remote_code=True)
    results = list(done.values())
    correct = sum((int(r['is_correct']) for r in results))
    t_start = time.time()
    for i, problem in enumerate(problems):
        pid = problem.id
        if pid in done:
            continue
        rec = solve(llm, tokenizer, problem.question, problem.answer, args.max_tokens)
        rec['problem_id'] = pid
        rec['ground_truth'] = problem.answer
        results.append(rec)
        correct += int(rec['is_correct'])
        n = len(results)
        avg_tok = sum((r['total_tokens'] for r in results)) / n
        think_rate = sum((any((s.get('has_think_tag') for s in r['samples'])) for r in results)) / n
        eta = (time.time() - t_start) / max(1, n - len(done)) * (len(problems) - n)
        print(f"[{n}/{len(problems)}] {pid} {('OK' if rec['is_correct'] else 'ERR')} tok={rec['total_tokens']} acc={correct / n:.1%} avg_tok={avg_tok:.0f} think_tag={think_rate:.1%} ETA={eta / 60:.0f}min", flush=True)
        if n % args.save_every == 0 or n == len(problems):
            os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
            json.dump(results, open(args.out, 'w'), indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    json.dump(results, open(args.out, 'w'), indent=2, ensure_ascii=False)
if __name__ == '__main__':
    main()
