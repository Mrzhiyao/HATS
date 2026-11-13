import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from vllm import LLM, SamplingParams
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_benchmark.datasets.loader import check_answer, load_dataset_by_name, normalize_answer
PROMPT_TPL = 'Please solve the following problem step by step. Show your reasoning and put your final answer in \\boxed{{}}.\n\nProblem: {question}'

def clean_decoded_answer(answer):
    if answer is None:
        return None
    return answer.replace('Ġ', ' ').replace('Ċ', '\n').strip()

def extract_final_answer(text):
    text_clean = text
    think_match = re.search('<think>.*?</think>', text, re.DOTALL)
    if think_match:
        text_clean = text[think_match.end():]
    matches = re.findall('\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}', text_clean)
    if matches:
        return clean_decoded_answer(matches[-1])
    matches = re.findall('\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}', text)
    return clean_decoded_answer(matches[-1]) if matches else None

def compute_logprob_mean(output):
    if not output.logprobs:
        return None
    values = []
    for token_logprobs in output.logprobs:
        if token_logprobs:
            chosen = next(iter(token_logprobs.values()))
            values.append(chosen.logprob if hasattr(chosen, 'logprob') else float(chosen))
    return sum(values) / len(values) if values else None

def parse_ok(answer):
    if not answer:
        return False
    try:
        from sympy import sympify
        from sympy.parsing.latex import parse_latex
        try:
            return parse_latex(answer) is not None
        except Exception:
            return sympify(answer) is not None
    except Exception:
        return False

def majority_vote(samples):
    valid = [(sample['answer'], normalize_answer(sample['answer'])) for sample in samples if sample.get('answer')]
    if not valid:
        return (samples[0].get('answer'), '', 0, 0)
    counts = Counter((normalized for _, normalized in valid))
    top, top_n = counts.most_common(1)[0]
    final = next((answer for answer, normalized in valid if normalized == top))
    return (final, top, top_n, len(valid))

def finish_result(difficulty, mode, stages, samples, elapsed_s, gt, override_answer=None):
    final = override_answer if override_answer is not None else majority_vote(samples)[0]
    _, _, top_n, n_valid = majority_vote(samples)
    return {'difficulty': difficulty, 'mode': mode, 'stages_used': stages, 'n_used': len(samples), 'majority_count': top_n, 'n_valid': n_valid, 'consensus': top_n / n_valid if n_valid else 0, 'samples': samples, 'total_tokens': sum((sample['tokens'] for sample in samples)), 'elapsed_s': elapsed_s, 'final_answer': final, 'is_correct': check_answer(final or '', gt)}

def hats_solve(llm, question, gt, difficulty, max_samples, k_stop_easy, k_stop_hard, easy_threshold, tau_low, tau_lp, abstain_token_frac, max_tokens):
    prompt = PROMPT_TPL.format(question=question)
    samples = []
    stages = ['S0_PDE', 'S1_GREEDY']
    t0 = time.time()
    greedy_params = SamplingParams(temperature=0, max_tokens=max_tokens, logprobs=1)
    greedy = llm.generate([prompt], greedy_params, use_tqdm=False)[0].outputs[0]
    greedy_answer = extract_final_answer(greedy.text)
    greedy_tokens = len(greedy.token_ids)
    lp_mean = compute_logprob_mean(greedy)
    samples.append({'answer': greedy_answer, 'tokens': greedy_tokens, 'hit_limit': greedy_tokens >= max_tokens - 8, 'lp_mean': lp_mean, 'is_greedy': True})
    length_abstain = greedy_tokens >= int(max_tokens * abstain_token_frac)
    can_commit_s1 = greedy_answer is not None and (not samples[-1]['hit_limit']) and (not length_abstain) and parse_ok(greedy_answer)
    if difficulty < tau_low and can_commit_s1:
        return finish_result(difficulty, 'S0+S1_easy', stages, samples, time.time() - t0, gt, override_answer=greedy_answer)
    if can_commit_s1 and lp_mean is not None and (lp_mean > tau_lp) and (difficulty < easy_threshold):
        return finish_result(difficulty, 'S1_high_conf', stages, samples, time.time() - t0, gt, override_answer=greedy_answer)
    stages.append('S2_SC')
    k_stop = k_stop_easy if difficulty < easy_threshold else k_stop_hard
    sample_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
    for _ in range(1, max_samples):
        output = llm.generate([prompt], sample_params, use_tqdm=False)[0].outputs[0]
        samples.append({'answer': extract_final_answer(output.text), 'tokens': len(output.token_ids), 'hit_limit': len(output.token_ids) >= max_tokens - 8, 'is_greedy': False})
        _, _, top_n, _ = majority_vote(samples)
        if top_n >= k_stop:
            return finish_result(difficulty, f'S2_majority_{top_n}', stages, samples, time.time() - t0, gt)
    return finish_result(difficulty, 'S2_max_budget', stages, samples, time.time() - t0, gt)

def load_done(path):
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        rows = json.load(f)
    return {row['problem_id']: row for row in rows}

def load_pde_predictions(path):
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, list):
        return {row['problem_id']: {'predicted_difficulty': row.get('difficulty', 0.5)} for row in raw if 'problem_id' in row}
    raise TypeError(f'Unsupported PDE prediction format in {path}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--pde_predictions', required=True)
    parser.add_argument('--dataset', default='math500')
    parser.add_argument('--out', required=True)
    parser.add_argument('--n_problems', type=int, default=2)
    parser.add_argument('--max_samples', type=int, default=8)
    parser.add_argument('--k_stop_easy', type=int, default=3)
    parser.add_argument('--k_stop_hard', type=int, default=4)
    parser.add_argument('--easy_threshold', type=float, default=0.5)
    parser.add_argument('--tau_low', type=float, default=0.1)
    parser.add_argument('--tau_lp', type=float, default=-0.4)
    parser.add_argument('--abstain_token_frac', type=float, default=0.8)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--max_model_len', type=int, default=5120)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    args = parser.parse_args()
    problems = load_dataset_by_name(args.dataset)[:args.n_problems]
    pde_pred = load_pde_predictions(args.pde_predictions)
    done = load_done(args.out)
    if done:
        print(f'Resuming {len(done)} done', flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    results = list(done.values())
    correct = sum((int(row['is_correct']) for row in results))
    for problem in problems:
        if problem.id in done:
            continue
        pred = pde_pred.get(problem.id, {})
        difficulty = pred.get('predicted_difficulty', 0.5)
        if difficulty < 0:
            difficulty = 0.5
        result = hats_solve(llm, problem.question, problem.answer, difficulty, args.max_samples, args.k_stop_easy, args.k_stop_hard, args.easy_threshold, args.tau_low, args.tau_lp, args.abstain_token_frac, args.max_tokens)
        result.update({'problem_id': problem.id, 'ground_truth': problem.answer, 'config': {'max_samples': args.max_samples, 'k_stop_easy': args.k_stop_easy, 'k_stop_hard': args.k_stop_hard, 'easy_threshold': args.easy_threshold, 'tau_low': args.tau_low, 'tau_lp': args.tau_lp, 'abstain_token_frac': args.abstain_token_frac, 'max_tokens': args.max_tokens}})
        results.append(result)
        correct += int(result['is_correct'])
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        n_done = len(results)
        avg_tokens = sum((row['total_tokens'] for row in results)) / n_done
        avg_samples = sum((row['n_used'] for row in results)) / n_done
        print(f"[{n_done}/{len(problems)}] {problem.id} {('OK' if result['is_correct'] else 'ERR')} d={difficulty:.2f} mode={result['mode']} n={result['n_used']} tok={result['total_tokens']} acc={correct / n_done:.1%} avg_n={avg_samples:.2f} avg_tok={avg_tokens:.0f}", flush=True)
    n_done = len(results)
    print(f"FINAL n={n_done} acc={correct / n_done:.4f} avg_samples={sum((row['n_used'] for row in results)) / n_done:.2f} avg_tokens={sum((row['total_tokens'] for row in results)) / n_done:.1f}", flush=True)
if __name__ == '__main__':
    main()
