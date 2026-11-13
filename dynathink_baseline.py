import json
import os
import sys
import time
import re
import argparse
from collections import Counter
from vllm import LLM, SamplingParams
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_benchmark.datasets.loader import load_dataset_by_name, normalize_answer, check_answer
SOLVE_TPL = 'Please solve the following problem step by step. Show your reasoning and put your final answer in \\boxed{{}}.\n\nProblem: {question}'
RATE_TPL = 'Rate the difficulty of solving the following math problem on a scale of 1 to 5, where 1 = trivial and 5 = competition-grade hard. Output only a single digit (1, 2, 3, 4, or 5).\n\nProblem: {question}\n\nDifficulty: '

def extract_final_answer(text):
    text_clean = text
    think_match = re.search('<think>.*?</think>', text, re.DOTALL)
    if think_match:
        text_clean = text[think_match.end():]
    matches = re.findall('\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}', text_clean)
    if matches:
        return matches[-1].strip()
    matches_all = re.findall('\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}', text)
    if matches_all:
        return matches_all[-1].strip()
    return None

def extract_rating(text):
    m = re.search('\\b([1-5])\\b', text)
    if m:
        return int(m.group(1))
    return 3

def majority_vote(samples):
    valid = [normalize_answer(s['answer']) for s in samples if s['answer']]
    if not valid:
        return (samples[0]['answer'], 0)
    cnt = Counter(valid)
    top, top_n = cnt.most_common(1)[0]
    final = next((s['answer'] for s in samples if s['answer'] and normalize_answer(s['answer']) == top))
    return (final, top_n)

def dynathink_solve(llm, question, gt, hard_threshold=3):
    samples = []
    total_tokens = 0
    t0 = time.time()
    rate_prompt = RATE_TPL.format(question=question)
    rate_params = SamplingParams(temperature=0, max_tokens=8)
    o = llm.generate([rate_prompt], rate_params, use_tqdm=False)[0].outputs[0]
    rating = extract_rating(o.text)
    rating_tokens = len(o.token_ids)
    total_tokens += rating_tokens
    solve_prompt = SOLVE_TPL.format(question=question)
    if rating <= hard_threshold:
        g_params = SamplingParams(temperature=0, max_tokens=4096)
        out = llm.generate([solve_prompt], g_params, use_tqdm=False)[0].outputs[0]
        ans = extract_final_answer(out.text)
        samples.append({'answer': ans, 'tokens': len(out.token_ids), 'hit_limit': len(out.token_ids) >= 4090})
        total_tokens += samples[-1]['tokens']
        final = ans
        n_used = 1
        mode = 'fast'
    else:
        g_params = SamplingParams(temperature=0, max_tokens=4096)
        out = llm.generate([solve_prompt], g_params, use_tqdm=False)[0].outputs[0]
        samples.append({'answer': extract_final_answer(out.text), 'tokens': len(out.token_ids), 'hit_limit': len(out.token_ids) >= 4090})
        total_tokens += samples[-1]['tokens']
        s_params = SamplingParams(temperature=0.7, max_tokens=4096, n=3)
        out2 = llm.generate([solve_prompt], s_params, use_tqdm=False)
        for o in out2[0].outputs:
            samples.append({'answer': extract_final_answer(o.text), 'tokens': len(o.token_ids), 'hit_limit': len(o.token_ids) >= 4090})
            total_tokens += samples[-1]['tokens']
        final, _ = majority_vote(samples)
        n_used = 4
        mode = 'slow'
    return {'rating': rating, 'rating_tokens': rating_tokens, 'mode': mode, 'n_used': n_used, 'samples': samples, 'total_tokens': total_tokens, 'elapsed_s': time.time() - t0, 'final_answer': final, 'is_correct': check_answer(final or '', gt)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    ap.add_argument('--dataset', default='math500')
    ap.add_argument('--out', default='results/dynathink_math500.json')
    ap.add_argument('--hard_threshold', type=int, default=3)
    args = ap.parse_args()
    print(f'Loading dataset {args.dataset}...')
    problems = load_dataset_by_name(args.dataset)
    print(f'  {len(problems)} problems')
    print(f'Loading vLLM model: {args.model}')
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=5120, gpu_memory_utilization=0.9, trust_remote_code=True)
    done = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            existing = json.load(f)
        done = {r['problem_id']: r for r in existing}
        print(f'Resuming: {len(done)} done')
    results = list(done.values())
    correct = sum((int(r['is_correct']) for r in results))
    t_start = time.time()
    new_done = 0
    for p in problems:
        pid = p.id
        if pid in done:
            continue
        result = dynathink_solve(llm, p.question, p.answer, args.hard_threshold)
        result['problem_id'] = pid
        result['ground_truth'] = p.answer
        results.append(result)
        correct += int(result['is_correct'])
        new_done += 1
        n_done = len(results)
        elapsed = time.time() - t_start
        speed = elapsed / new_done
        eta = speed * (len(problems) - n_done) / 60
        tag = 'OK' if result['is_correct'] else 'ERR'
        print(f"[{n_done}/{len(problems)}] {p.id}: {tag} r={result['rating']} mode={result['mode']} n={result['n_used']} tok={result['total_tokens']} | acc={correct / n_done:.1%} ETA={eta:.0f}min")
        if n_done % 10 == 0 or n_done == len(problems):
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    n = len(results)
    fast_n = sum((1 for r in results if r['mode'] == 'fast'))
    slow_n = n - fast_n
    print(f"\n{'=' * 60}")
    print(f'DynaThink — {args.dataset.upper()}')
    print(f"{'=' * 60}")
    print(f'Acc: {correct}/{n} ({correct / n:.1%})')
    print(f'Routing: fast={fast_n} slow={slow_n}')
    total_tokens = sum((r['total_tokens'] for r in results))
    print(f'Total tokens: {total_tokens}  (avg {total_tokens / n:.0f}/problem)')
if __name__ == '__main__':
    main()
