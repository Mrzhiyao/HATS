import json
import os
import sys
import time
import math
import argparse
from collections import Counter
from vllm import LLM, SamplingParams
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_benchmark.datasets.loader import load_dataset_by_name, normalize_answer, check_answer
PROMPT_TPL = 'Please solve the following problem step by step. Show your reasoning and put your final answer in \\boxed{{}}.\n\nProblem: {question}'

def extract_final_answer(text):
    import re
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

def compute_logprob_mean(output):
    if not output.logprobs:
        return None
    lps = []
    for tok_lps in output.logprobs:
        if tok_lps:
            chosen = next(iter(tok_lps.values()))
            lp = chosen.logprob if hasattr(chosen, 'logprob') else float(chosen)
            lps.append(lp)
    return sum(lps) / len(lps) if lps else None

def majority_vote(samples):
    valid = [(s['answer'], normalize_answer(s['answer'])) for s in samples if s['answer']]
    if not valid:
        return (samples[0]['answer'], '', 0, 0)
    cnt = Counter([n for _, n in valid])
    top, top_n = cnt.most_common(1)[0]
    final = next((a for a, n in valid if n == top))
    return (final, top, top_n, len(valid))

def parse_ok(answer):
    if not answer:
        return False
    try:
        from sympy.parsing.latex import parse_latex
        from sympy import sympify
        try:
            return parse_latex(answer) is not None
        except Exception:
            try:
                return sympify(answer) is not None
            except Exception:
                return False
    except Exception:
        return False

def msap_solve(llm, question, gt, difficulty, tau_low=0.3, tau_lp=-0.4, max_samples=4):
    prompt = PROMPT_TPL.format(question=question)
    samples = []
    total_tokens = 0
    stages_used = []
    t0 = time.time()
    stages_used.append('S0_PDE')
    if difficulty < tau_low:
        pass
    stages_used.append('S1_GREEDY')
    g_params = SamplingParams(temperature=0, max_tokens=4096, logprobs=1)
    o = llm.generate([prompt], g_params, use_tqdm=False)[0].outputs[0]
    greedy_text = o.text
    lp_mean = compute_logprob_mean(o)
    greedy_ans = extract_final_answer(greedy_text)
    samples.append({'answer': greedy_ans, 'tokens': len(o.token_ids), 'hit_limit': len(o.token_ids) >= 4090, 'lp_mean': lp_mean, 'is_greedy': True})
    total_tokens += samples[-1]['tokens']
    can_commit_s1 = greedy_ans is not None and (not samples[-1]['hit_limit']) and parse_ok(greedy_ans)
    if difficulty < tau_low and can_commit_s1:
        return _result(difficulty, 'S0+S1_easy', stages_used, samples, total_tokens, time.time() - t0, gt, greedy_ans)
    if can_commit_s1 and lp_mean is not None and (lp_mean > tau_lp) and (difficulty < 0.5):
        return _result(difficulty, 'S1_high_conf', stages_used, samples, total_tokens, time.time() - t0, gt, greedy_ans)
    stages_used.append('S2_SC')
    k_stop = 2 if difficulty < 0.5 else 3
    s_params = SamplingParams(temperature=0.7, max_tokens=4096)
    for i in range(1, max_samples):
        o = llm.generate([prompt], s_params, use_tqdm=False)[0].outputs[0]
        ans = extract_final_answer(o.text)
        samples.append({'answer': ans, 'tokens': len(o.token_ids), 'hit_limit': len(o.token_ids) >= 4090, 'is_greedy': False})
        total_tokens += samples[-1]['tokens']
        _, _, top_n, _ = majority_vote(samples)
        if top_n >= k_stop:
            return _result(difficulty, f'S2_majority_{top_n}', stages_used, samples, total_tokens, time.time() - t0, gt)
    return _result(difficulty, 'S2_max_budget', stages_used, samples, total_tokens, time.time() - t0, gt)

def _result(difficulty, mode, stages, samples, total_tokens, elapsed, gt, override_answer=None):
    if override_answer is not None:
        final = override_answer
    else:
        final, _, _, _ = majority_vote(samples)
    ok = check_answer(final or '', gt)
    return {'difficulty': difficulty, 'mode': mode, 'stages_used': stages, 'n_used': len(samples), 'samples': samples, 'total_tokens': total_tokens, 'elapsed_s': elapsed, 'final_answer': final, 'is_correct': ok}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    ap.add_argument('--pde_predictions', required=True, help='JSON {pid: {predicted_difficulty: float, ...}}')
    ap.add_argument('--dataset', default='math500')
    ap.add_argument('--out', default='results/hats_math500.json')
    ap.add_argument('--tau_low', type=float, default=0.3)
    ap.add_argument('--tau_lp', type=float, default=-0.4)
    ap.add_argument('--max_samples', type=int, default=4)
    args = ap.parse_args()
    print(f'Loading dataset {args.dataset}...')
    problems = load_dataset_by_name(args.dataset)
    print(f'  {len(problems)} problems')
    print(f'Loading PDE predictions from {args.pde_predictions}...')
    with open(args.pde_predictions) as f:
        pde_pred = json.load(f)
    print(f'  {len(pde_pred)} predictions loaded')
    print(f'Loading vLLM model: {args.model}')
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=5120, gpu_memory_utilization=0.9, trust_remote_code=True)
    print('Model loaded!')
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
        if pid not in pde_pred:
            difficulty = 0.5
        else:
            difficulty = pde_pred[pid].get('predicted_difficulty', 0.5)
            if difficulty < 0:
                difficulty = 0.5
        result = msap_solve(llm, p.question, p.answer, difficulty, tau_low=args.tau_low, tau_lp=args.tau_lp, max_samples=args.max_samples)
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
        print(f"[{n_done}/{len(problems)}] {pid}: {tag} d={difficulty:.2f} mode={result['mode']} n={result['n_used']} tok={result['total_tokens']} | acc={correct / n_done:.1%} ETA={eta:.0f}min")
        if n_done % 10 == 0 or n_done == len(problems):
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    n = len(results)
    total_calls = sum((r['n_used'] for r in results))
    total_tokens = sum((r['total_tokens'] for r in results))
    routes = Counter((r['mode'] for r in results))
    print(f"\n{'=' * 70}")
    print(f'HATS — {args.dataset.upper()}')
    print(f"{'=' * 70}")
    print(f'Accuracy: {correct}/{n} ({correct / n:.1%})')
    print(f'Total calls: {total_calls}  (avg {total_calls / n:.2f}/problem)')
    print(f'Total tokens: {total_tokens}  (avg {total_tokens / n:.0f}/problem)')
    print(f'Routing: {dict(routes)}')
if __name__ == '__main__':
    main()
