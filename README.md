# HATS

This is an implementation of our paper Prefill-Guided Trace Allocation for Sample-Efficient Test-Time Scaling, accepted at the Conference on Neural Information Processing Systems (NeurIPS), 2026. In this repo, we implement the proposed Hidden-State Adaptive Trace Scheduler (HATS), which predicts greedy-trace reliability from prefill hidden states and routes each problem to one checked trace, a small vote, or the full self-consistency budget.

The repository contains the code and small JSON files needed to run HATS and the main baselines. Plotting scripts, model checkpoints, and large third-party datasets are not included.

## Files

- `hats.py`: HATS/MSAP with SC-4 cap.
- `hats_k8_pilot.py`: HATS variant with a larger sample cap.
- `collect_prefill_hidden.py`: extracts prefill hidden-state features for PDE training.
- `train_pde.py`: trains the prefill difficulty estimator and exports predictions.
- `sc4_baseline_vllm.py`: SC-4 baseline.
- `sc_k_baseline_vllm.py`: generic SC-k baseline.
- `baseline_real_run.py`: ESC, AC, TVS, DSC-style, and RPC-style online baselines.
- `dsc_prior.py`: prompt-feature prior used by the DSC-style baseline.
- `dynathink_baseline.py`: DynaThink-style self-rating baseline.
- `qwen3_nothink_sc4.py`: Qwen3 no-thinking SC-4 helper.
- `tts_benchmark/datasets/loader.py`: dataset loading and answer checking utilities.

## Included Data

- `data/pde_math500_predictions.json`: PDE difficulty scores for the main MATH-500 HATS run.
- `data/pde_math500_oof_predictions.json`: strict out-of-fold MATH-500 PDE scores reconstructed from the OOF run.
- `data/pde_gsm8k_predictions.json`: PDE scores for GSM8K transfer checks.
- `data/pde_aime_predictions.json`: PDE scores for AIME transfer checks.
- `data/sc4_baseline_math500.json`: SC-4 trace pool used for PDE labels and baseline reference checks.
- `data/hats_math500_output.json`: main MATH-500 HATS output for quick verification.
- `data/hats_math500_oof_output.json`: strict OOF MATH-500 HATS output for quick verification.
- `data/dsc_prior_math500.json`: prompt-feature prior for the DSC-style baseline.

## Reproducibility Scope

This package supports the main HATS MATH-500 run and the online baseline runners used in the paper. The JSON files in `data/` are generated intermediate artifacts, included so the main HATS command can be checked without rerunning all feature extraction and GPU generation.

The package does not include plotting scripts, figure files, remote execution logs, hardware profiling traces, model checkpoints, or full experiment logs.

## Environment

The experiments use Python 3.11, CUDA-compatible PyTorch, and vLLM. The default model is `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`. A single 24 GB GPU is sufficient for the MATH-500 settings used here.

## Setup

```bash
pip install -r requirements.txt
```

The scripts default to HuggingFace model identifiers. For local model storage, pass `--model /path/to/model`.

## Raw Data Access

Raw datasets are loaded with HuggingFace `datasets`:

- MATH-500: `HuggingFaceH4/MATH-500`
- GSM8K: `gsm8k`, config `main`
- AIME 2022--2024: `AI-MO/aimo-validation-aime`

The loader code is in `tts_benchmark/datasets/loader.py`.

The additional zero-shot transfer benchmarks are taken from:

- SVAMP: `ChilleD/SVAMP`, test split, 300 problems
- GSM-Hard: `reasoning-machines/gsm-hard`, 300 problems sampled with random seed 42
- MinervaMath: `math-ai/minervamath`, test split, 272 problems
- AMC23: `math-ai/amc23`, test split, 40 problems
- AIME 2025: `math-ai/aime25`, test split, 30 problems

## Typical Workflow

1. Run SC-4 to produce greedy/sample traces for labels:

```bash
python sc4_baseline_vllm.py --dataset math500 --out results/sc4_math500.json
```

2. Collect prefill features:

```bash
python collect_prefill_hidden.py --dataset math500 --out results/prefill_math500.npz
```

3. Train PDE and export predictions:

```bash
python train_pde.py --features results/prefill_math500.npz --sc_results results/sc4_math500.json --out results/pde_math500.pt --predictions_out results/pde_math500.json
```

4. Run HATS using the included PDE scores:

```bash
python hats.py --dataset math500 --pde_predictions data/pde_math500_predictions.json --out results/hats_math500.json
```

5. Run online baselines:

```bash
python baseline_real_run.py --dataset math500 --policy ac --output results/ac_math500.json
python baseline_real_run.py --dataset math500 --policy esc --output results/esc_math500.json
python baseline_real_run.py --dataset math500 --policy tvs --output results/tvs_math500.json
python baseline_real_run.py --dataset math500 --policy dsc --dsc_prior_path data/dsc_prior_math500.json --output results/dsc_math500.json
python baseline_real_run.py --dataset math500 --policy rpc --output results/rpc_math500.json
```

## Citation

```bibtex
@inproceedings{yao2026hats,
  title     = {Prefill-Guided Trace Allocation for Sample-Efficient Test-Time Scaling},
  author    = {Yao, Zhi and Tang, Zhiqing and Cui, Hanshuai and Ma, Qianli and Meng, Fanshuai and Jia, Weijia},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

This project is released under the MIT License. See `LICENSE` for details.
