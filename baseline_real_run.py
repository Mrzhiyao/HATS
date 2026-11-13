from __future__ import annotations
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from vllm import LLM, SamplingParams
from tts_benchmark.datasets.loader import load_dataset_by_name, normalize_answer, check_answer
PROMPT_TPL = 'Please solve the following problem step by step. Show your reasoning and put your final answer in \\boxed{{}}.\n\nProblem: {question}'

def extract_final_answer(text: str) -> str | None:
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

def stop_esc(samples: list[dict], window: int) -> bool:
    if len(samples) < window:
        return False
    last = samples[-window:]
    norms = [normalize_answer(s['answer']) for s in last if s.get('answer')]
    if len(norms) < window:
        return False
    return len(set(norms)) == 1

def _beta_sf_at_half(a: int, b: int) -> float:
    import math
    n = a + b - 1
    half_pow = 0.5 ** n
    return sum((math.comb(n, j) for j in range(a))) * half_pow

def stop_ac(samples: list[dict], threshold: float, n_min: int=2) -> bool:
    if len(samples) < n_min:
        return False
    norms = [normalize_answer(s['answer']) for s in samples if s.get('answer')]
    if not norms:
        return False
    cnt = Counter(norms)
    _, top_n = cnt.most_common(1)[0]
    rest = len(norms) - top_n
    p_top = _beta_sf_at_half(top_n + 1, rest + 1)
    return p_top >= threshold

def stop_tvs(samples: list[dict], k: int, n_min: int=1) -> bool:
    if len(samples) < n_min:
        return False
    norms = [normalize_answer(s['answer']) for s in samples if s.get('answer')]
    if not norms:
        return False
    cnt = Counter(norms)
    _, top_n = cnt.most_common(1)[0]
    return top_n >= k

def _rpc_active_samples(samples: list[dict], prune_quantile: float) -> list[dict]:
    if prune_quantile <= 0.0:
        return list(samples)
    scored = [s for s in samples if s.get('mean_logprob') is not None]
    if len(scored) < 2:
        return list(samples)
    cut_count = int(round(prune_quantile * len(scored)))
    if cut_count <= 0:
        return list(samples)
    cut_count = min(cut_count, len(scored) - 1)
    dropped = set((id(s) for s in sorted(scored, key=lambda s: s['mean_logprob'])[:cut_count]))
    return [s for s in samples if id(s) not in dropped]

def _rpc_weighted_top_fraction(samples: list[dict]) -> tuple[float, str | None]:
    weights: dict[str, float] = {}
    raw_answers: dict[str, str] = {}
    total = 0.0
    for s in samples:
        ans = s.get('answer')
        if not ans:
            continue
        norm = normalize_answer(ans)
        if not norm:
            continue
        mlp = s.get('mean_logprob')
        import math
        w = math.exp(mlp) if isinstance(mlp, (int, float)) else 1.0
        weights[norm] = weights.get(norm, 0.0) + w
        raw_answers.setdefault(norm, ans)
        total += w
    if total <= 0.0 or not weights:
        return (0.0, None)
    top_norm, top_w = max(weights.items(), key=lambda kv: kv[1])
    return (top_w / total, raw_answers[top_norm])

def stop_rpc(samples: list[dict], threshold: float, n_min: int, prune_quantile: float) -> bool:
    if len(samples) < n_min:
        return False
    kept = _rpc_active_samples(samples, prune_quantile)
    if not kept:
        return False
    frac, _ = _rpc_weighted_top_fraction(kept)
    return frac >= threshold

def stop_dsc(samples: list[dict], threshold: float, n_min: int, alpha0: float, beta0: float) -> bool:
    if len(samples) < n_min:
        return False
    norms = [normalize_answer(s['answer']) for s in samples if s.get('answer')]
    if not norms:
        return False
    cnt = Counter(norms)
    _, top_n = cnt.most_common(1)[0]
    rest = len(norms) - top_n
    a = int(round(alpha0 + top_n))
    b = int(round(beta0 + rest))
    a, b = (max(a, 1), max(b, 1))
    return _beta_sf_at_half(a, b) >= threshold

def make_stop_fn(policy: str, args, *, dsc_prior_alpha: float=1.0, dsc_prior_beta: float=1.0):
    if policy == 'esc':
        return lambda samples: stop_esc(samples, args.window)
    if policy == 'ac':
        return lambda samples: stop_ac(samples, args.threshold, args.n_min)
    if policy == 'tvs':
        return lambda samples: stop_tvs(samples, args.k, args.n_min)
    if policy == 'dsc':
        return lambda samples: stop_dsc(samples, args.threshold, args.n_min, alpha0=dsc_prior_alpha, beta0=dsc_prior_beta)
    if policy == 'rpc':
        return lambda samples: stop_rpc(samples, args.threshold, args.n_min, args.rpc_prune_quantile)
    raise ValueError(f'unknown policy: {policy}')

def majority_vote(samples: list[dict]) -> str:
    valid = [s for s in samples if s.get('answer')]
    if not valid:
        return samples[0].get('answer', '') if samples else ''
    norms = [normalize_answer(s['answer']) for s in valid]
    cnt = Counter(norms)
    best, _ = cnt.most_common(1)[0]
    for s in valid:
        if normalize_answer(s['answer']) == best:
            return s['answer']
    return valid[0]['answer']

def rpc_final_answer(samples: list[dict], prune_quantile: float) -> str:
    kept = _rpc_active_samples(samples, prune_quantile)
    if not kept:
        return majority_vote(samples)
    _, ans = _rpc_weighted_top_fraction(kept)
    if ans is None:
        return majority_vote(samples)
    return ans

def _mean_logprob_from_vllm_output(output) -> float | None:
    lp = getattr(output, 'logprobs', None)
    if not lp:
        return None
    total = 0.0
    n = 0
    for step in lp:
        if not step:
            continue
        try:
            best = max(step.values(), key=lambda x: getattr(x, 'logprob', x))
            val = getattr(best, 'logprob', best)
        except Exception:
            continue
        total += float(val)
        n += 1
    if n == 0:
        return None
    return total / n

def solve_problem(llm: LLM, prompt: str, gt: str, stop_fn, n_max: int, max_tokens: int, greedy_first: bool, *, with_logprobs: bool=False, final_picker=None) -> dict:
    samples: list[dict] = []
    total_tokens = 0
    t_start = time.time()
    for k in range(1, n_max + 1):
        sp_kwargs = dict(max_tokens=max_tokens)
        if with_logprobs:
            sp_kwargs['logprobs'] = 1
        if greedy_first and k == 1:
            params = SamplingParams(temperature=0.0, **sp_kwargs)
        else:
            params = SamplingParams(temperature=0.7, top_p=0.95, **sp_kwargs)
        t0 = time.time()
        out = llm.generate([prompt], params, use_tqdm=False)
        out_obj = out[0].outputs[0]
        text = out_obj.text
        tokens = len(out_obj.token_ids)
        ans = extract_final_answer(text)
        rec = {'answer': ans, 'tokens': tokens, 'hit_limit': tokens >= max_tokens - 6, 'time_ms': int((time.time() - t0) * 1000)}
        if with_logprobs:
            rec['mean_logprob'] = _mean_logprob_from_vllm_output(out_obj)
        samples.append(rec)
        total_tokens += tokens
        if stop_fn(samples):
            break
    final = final_picker(samples) if final_picker is not None else majority_vote(samples)
    return {'n_used': len(samples), 'samples': samples, 'final_answer': final, 'is_correct': check_answer(final, gt), 'total_tokens': total_tokens, 'wall_ms': int((time.time() - t_start) * 1000)}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--policy', required=True, choices=['esc', 'ac', 'tvs', 'dsc', 'rpc'])
    ap.add_argument('--dataset', required=True, help='dataset name accepted by load_dataset_by_name (e.g. math500, gsm8k, aime2024, aime)')
    ap.add_argument('--model_path', default=os.environ.get('MODEL_PATH', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B'))
    ap.add_argument('--output', required=True, help='path to per-problem JSON results file')
    ap.add_argument('--n_problems', type=int, default=None, help='cap problem count for debugging; None = all')
    ap.add_argument('--n_max', type=int, default=16, help='hard cap on samples per problem')
    ap.add_argument('--max_tokens', type=int, default=4096)
    ap.add_argument('--max_model_len', type=int, default=5120)
    ap.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    ap.add_argument('--greedy_first', action='store_true', help='first sample uses temperature 0; default off, matches original ESC/AC/TVS papers')
    ap.add_argument('--window', type=int, default=2, help='ESC sliding window size')
    ap.add_argument('--threshold', type=float, default=0.95, help='AC Beta posterior threshold')
    ap.add_argument('--k', type=int, default=2, help='TVS vote count threshold')
    ap.add_argument('--n_min', type=int, default=2, help='minimum samples before allowing stop (applies to AC, TVS and DSC)')
    ap.add_argument('--dsc_prior_path', default=None, help='path to a trained DSCPrior JSON (required for --policy dsc)')
    ap.add_argument('--dsc_prior_lambda', type=float, default=4.0, help='DSC: pseudo-count strength of the easiness prior; alpha0 = 1 + lambda * e, beta0 = 1 + lambda * (1-e)')
    ap.add_argument('--rpc_prune_quantile', type=float, default=0.2, help='RPC: drop samples whose mean logprob is below this quantile of the per-stream distribution')
    ap.add_argument('--save_every', type=int, default=10, help='checkpoint every N completed problems')
    args = ap.parse_args()
    done: dict[str, dict] = {}
    if os.path.exists(args.output):
        with open(args.output) as f:
            existing = json.load(f)
        done = {r['problem_id']: r for r in existing}
        print(f'[resume] {len(done)} problems already in {args.output}')
    problems = load_dataset_by_name(args.dataset)
    if args.n_problems:
        problems = problems[:args.n_problems]
    print(f'[init] policy={args.policy} dataset={args.dataset} n_problems={len(problems)} n_max={args.n_max} max_tokens={args.max_tokens}')
    print(f'[init] hyperparams: window={args.window} threshold={args.threshold} k={args.k} n_min={args.n_min} greedy_first={args.greedy_first}')
    print(f'[init] loading vLLM model: {args.model_path}')
    llm = LLM(model=args.model_path, tensor_parallel_size=1, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    print('[init] model loaded.')
    dsc_prior = None
    if args.policy == 'dsc':
        if not args.dsc_prior_path:
            raise SystemExit('--policy dsc requires --dsc_prior_path')
        from dsc_prior import DSCPrior
        dsc_prior = DSCPrior.load(args.dsc_prior_path)
        print(f'[init] DSC prior loaded from {args.dsc_prior_path} (lambda={args.dsc_prior_lambda})')
    results = list(done.values())
    correct = sum((int(r.get('is_correct', False)) for r in results))
    samples_used = sum((r.get('n_used', 0) for r in results))
    tokens_used = sum((r.get('total_tokens', 0) for r in results))
    t_start = time.time()
    initial_done = len(done)
    for i, p in enumerate(problems):
        pid = f'{args.dataset}_{i}'
        if pid in done:
            continue
        prompt = PROMPT_TPL.format(question=p.question)
        if args.policy == 'dsc':
            e = dsc_prior.predict(p.question)
            alpha0 = 1.0 + args.dsc_prior_lambda * e
            beta0 = 1.0 + args.dsc_prior_lambda * (1.0 - e)
            stop_fn = make_stop_fn(args.policy, args, dsc_prior_alpha=alpha0, dsc_prior_beta=beta0)
            extra = {'dsc_easiness': e, 'dsc_alpha0': alpha0, 'dsc_beta0': beta0}
        else:
            stop_fn = make_stop_fn(args.policy, args)
            extra = {}
        with_logprobs = args.policy == 'rpc'
        final_picker = (lambda samples: rpc_final_answer(samples, args.rpc_prune_quantile)) if args.policy == 'rpc' else None
        out = solve_problem(llm, prompt, p.answer, stop_fn=stop_fn, n_max=args.n_max, max_tokens=args.max_tokens, greedy_first=args.greedy_first, with_logprobs=with_logprobs, final_picker=final_picker)
        out['problem_id'] = pid
        out['ground_truth'] = p.answer
        out['policy'] = args.policy
        out.update(extra)
        results.append(out)
        correct += int(out['is_correct'])
        samples_used += out['n_used']
        tokens_used += out['total_tokens']
        n_done = len(results)
        new_done = n_done - initial_done
        elapsed = time.time() - t_start
        per_q = elapsed / max(new_done, 1)
        eta_min = per_q * (len(problems) - n_done) / 60.0
        tag = 'OK' if out['is_correct'] else 'ERR'
        print(f"[{n_done}/{len(problems)}] {pid}: {tag} n={out['n_used']} tok={out['total_tokens']} wall={out['wall_ms']}ms | acc={correct / n_done:.1%} avg_n={samples_used / n_done:.2f} avg_tok={tokens_used / n_done:.0f} ETA={eta_min:.0f}min", flush=True)
        if n_done % args.save_every == 0 or n_done == len(problems):
            os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    n = len(results)
    print(f"\n{'=' * 70}")
    print(f'{args.policy.upper()} REAL-RUN — {args.dataset} RESULTS')
    print(f"{'=' * 70}")
    print(f'Accuracy: {correct}/{n} ({correct / n:.1%})')
    print(f'Avg samples per problem: {samples_used / n:.2f}')
    print(f'Avg tokens per problem: {tokens_used / n:.0f}')
    print(f'Total tokens: {tokens_used}')
    print(f'Saved to {args.output}')
if __name__ == '__main__':
    main()
