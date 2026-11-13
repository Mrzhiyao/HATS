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

def majority_vote(samples):
    valid = [(sample['answer'], normalize_answer(sample['answer'])) for sample in samples if sample.get('answer')]
    if not valid:
        return (samples[0].get('answer'), '', 0, 0)
    counts = Counter((normalized for _, normalized in valid))
    top, top_n = counts.most_common(1)[0]
    final = next((answer for answer, normalized in valid if normalized == top))
    return (final, top, top_n, len(valid))

def solve(llm, question, gt, k, max_tokens):
    prompt = PROMPT_TPL.format(question=question)
    samples = []
    start_time = time.time()
    greedy_params = SamplingParams(temperature=0, max_tokens=max_tokens)
    greedy_output = llm.generate([prompt], greedy_params, use_tqdm=False)[0].outputs[0]
    samples.append({'answer': extract_final_answer(greedy_output.text), 'tokens': len(greedy_output.token_ids), 'hit_limit': len(greedy_output.token_ids) >= max_tokens - 8, 'is_greedy': True})
    if k > 1:
        sample_params = SamplingParams(temperature=0.7, max_tokens=max_tokens, n=k - 1)
        sampled_outputs = llm.generate([prompt], sample_params, use_tqdm=False)[0].outputs
        for output in sampled_outputs:
            samples.append({'answer': extract_final_answer(output.text), 'tokens': len(output.token_ids), 'hit_limit': len(output.token_ids) >= max_tokens - 8, 'is_greedy': False})
    final, _, top_n, n_valid = majority_vote(samples)
    return {'n_samples': k, 'final_answer': final, 'is_correct': check_answer(final or '', gt), 'majority_count': top_n, 'n_valid': n_valid, 'consensus': top_n / n_valid if n_valid else 0, 'total_tokens': sum((sample['tokens'] for sample in samples)), 'elapsed_s': time.time() - start_time, 'samples': samples}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B')
    parser.add_argument('--dataset', default='math500')
    parser.add_argument('--k', type=int, default=8)
    parser.add_argument('--n_problems', type=int, default=2)
    parser.add_argument('--out', required=True)
    parser.add_argument('--max_tokens', type=int, default=4096)
    parser.add_argument('--max_model_len', type=int, default=5120)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9)
    args = parser.parse_args()
    problems = load_dataset_by_name(args.dataset)[:args.n_problems]
    done = {}
    if os.path.exists(args.out):
        done = {row['problem_id']: row for row in json.load(open(args.out))}
        print(f'Resuming {len(done)} done', flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization, trust_remote_code=True)
    results = list(done.values())
    correct = sum((int(row['is_correct']) for row in results))
    for problem in problems:
        pid = problem.id
        if pid in done:
            continue
        result = solve(llm, problem.question, problem.answer, args.k, args.max_tokens)
        result.update({'problem_id': pid, 'ground_truth': problem.answer})
        results.append(result)
        correct += int(result['is_correct'])
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        json.dump(results, open(args.out, 'w'), indent=2, ensure_ascii=False)
        n_done = len(results)
        avg_tokens = sum((row['total_tokens'] for row in results)) / n_done
        print(f"[{n_done}/{len(problems)}] {pid} {('OK' if result['is_correct'] else 'ERR')} cons={result['consensus']:.2f} tok={result['total_tokens']} acc={correct / n_done:.1%} avg_tok={avg_tokens:.0f}", flush=True)
    n_done = len(results)
    print(f"FINAL n={n_done} acc={correct / n_done:.4f} avg_tokens={sum((row['total_tokens'] for row in results)) / n_done:.1f}", flush=True)
if __name__ == '__main__':
    main()
