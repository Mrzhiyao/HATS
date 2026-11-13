import argparse, json, os, sys, time, re
from collections import Counter
from vllm import LLM, SamplingParams
from tts_benchmark.datasets.loader import load_dataset_by_name, normalize_answer, check_answer
MODEL_PATH = os.environ.get('MODEL_PATH', 'deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
PROMPT_TPL = 'Please solve the following problem step by step. Show your reasoning and put your final answer in \\boxed{{}}.\n\nProblem: {question}'

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

def sc4_solve(llm, question, gt):
    prompt = PROMPT_TPL.format(question=question)
    samples = []
    total_tokens = 0
    greedy_params = SamplingParams(temperature=0, max_tokens=4096)
    sample_params = SamplingParams(temperature=0.7, max_tokens=4096, n=3)
    t0 = time.time()
    out = llm.generate([prompt], greedy_params, use_tqdm=False)
    text = out[0].outputs[0].text
    tokens = len(out[0].outputs[0].token_ids)
    answer = extract_final_answer(text)
    samples.append({'answer': answer, 'tokens': tokens, 'hit_limit': tokens >= 4090})
    total_tokens += tokens
    out2 = llm.generate([prompt], sample_params, use_tqdm=False)
    for o in out2[0].outputs:
        text = o.text
        tokens = len(o.token_ids)
        answer = extract_final_answer(text)
        samples.append({'answer': answer, 'tokens': tokens, 'hit_limit': tokens >= 4090})
        total_tokens += tokens
    dt = time.time() - t0
    normalized = [normalize_answer(s['answer']) for s in samples]
    valid = [n for n in normalized if n]
    if valid:
        counts = Counter(valid)
        best_ans, best_cnt = counts.most_common(1)[0]
        consensus = best_cnt / len(valid)
        final = next((s['answer'] for s in samples if s['answer'] and normalize_answer(s['answer']) == best_ans))
    else:
        best_ans = ''
        consensus = 0
        final = samples[0]['answer']
    return {'n_samples': 4, 'final_answer': final, 'is_correct': check_answer(final, gt), 'consensus': consensus, 'total_tokens': total_tokens, 'time_ms': int(dt * 1000), 'samples': samples}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=MODEL_PATH)
    ap.add_argument('--dataset', default='math500')
    ap.add_argument('--out', default='results/sc4_baseline_math500.json')
    ap.add_argument('--n_problems', type=int, default=None)
    args = ap.parse_args()
    problems = load_dataset_by_name(args.dataset)
    if args.n_problems:
        problems = problems[:args.n_problems]
    output_file = args.out
    done = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            existing = json.load(f)
        done = {r['problem_id']: r for r in existing}
        print(f'Resuming: {len(done)} problems already done')
    print(f'Loading vLLM model: {args.model}')
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=5120, gpu_memory_utilization=0.9, trust_remote_code=True)
    print('Model loaded!')
    results = list(done.values())
    correct = sum((r['is_correct'] for r in results))
    t_start = time.time()
    for i, p in enumerate(problems):
        pid = p.id
        if pid in done:
            continue
        result = sc4_solve(llm, p.question, p.answer)
        result['problem_id'] = pid
        result['ground_truth'] = p.answer
        results.append(result)
        correct += result['is_correct']
        n_done = len(results)
        elapsed = time.time() - t_start
        new_done = n_done - len(done)
        speed = elapsed / new_done if new_done > 0 else 0
        remaining = len(problems) - n_done
        eta_min = speed * remaining / 60
        tag = 'OK' if result['is_correct'] else 'ERR'
        print(f"[{n_done}/{len(problems)}] {pid}: {tag} cons={result['consensus']:.2f} tok={result['total_tokens']} | acc={correct / n_done:.1%} ETA={eta_min:.0f}min")
        if n_done % 10 == 0 or n_done == len(problems):
            if os.path.dirname(output_file):
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
    if os.path.dirname(output_file):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    n = len(results)
    total_calls = n * 4
    total_tokens = sum((r['total_tokens'] for r in results))
    print(f"\n{'=' * 70}")
    print(f'SC-4 BASELINE — {args.dataset.upper()} RESULTS')
    print(f"{'=' * 70}")
    print(f'Accuracy: {correct}/{n} ({correct / n:.1%})')
    print(f'Total calls: {total_calls}')
    print(f'Total tokens: {total_tokens}')
    print(f'Avg tokens per problem: {total_tokens / n:.0f}')
    print(f'Saved to {output_file}')
if __name__ == '__main__':
    main()
