import json
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
from datasets import load_dataset

@dataclass
class Problem:
    id: str
    question: str
    answer: str
    raw_answer: str
    difficulty: str = ''
    subject: str = ''
    source: str = ''

def extract_boxed_answer(text: str) -> str:
    pattern = '\\\\boxed\\{([^{}]*(?:\\{[^{}]*\\}[^{}]*)*)\\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return text.strip()

def _try_eval_frac(s: str) -> float | None:
    import re as _re
    m = _re.match('\\\\frac\\{([^}]+)\\}\\{([^}]+)\\}', s)
    if m:
        try:
            return float(m.group(1)) / float(m.group(2))
        except (ValueError, ZeroDivisionError):
            pass
    return None

def normalize_answer(answer) -> str:
    if answer is None:
        return ''
    answer = str(answer).strip()
    answer = answer.replace('\\$', '').replace('$', '')
    answer = answer.replace('\\%', '%')
    answer = answer.replace(' ', '')
    answer = answer.replace('^\\circ', '').replace('°', '').replace('\\circ', '')
    answer = answer.replace('\\left(', '(').replace('\\right)', ')')
    answer = answer.replace('\\left[', '[').replace('\\right]', ']')
    answer = answer.replace('\\left\\{', '{').replace('\\right\\}', '}')
    answer = answer.replace('\\dfrac', '\\frac')
    answer = answer.replace('\\tfrac', '\\frac')
    answer = answer.replace('\\text{', '').replace('\\mathrm{', '').replace('\\mathbf{', '')
    while answer.endswith('}') and answer.count('{') < answer.count('}'):
        answer = answer[:-1]
    answer = answer.replace('\\,', '').replace('\\;', '').replace('\\!', '')
    answer = answer.replace('\\cdot', '*')
    try:
        val = float(answer)
        if val == int(val):
            answer = str(int(val))
        else:
            answer = str(val)
    except (ValueError, OverflowError):
        pass
    return answer

def _normalize_set_answer(s: str) -> str | None:
    import re as _re
    parts = _re.split('[,;\\s]+', s.strip())
    nums = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        try:
            nums.append(float(p))
        except ValueError:
            return None
    if len(nums) >= 2:
        return ','.join((str(n) for n in sorted(nums)))
    return None

def check_answer(predicted: str, ground_truth: str) -> bool:
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    if pred_norm == gt_norm:
        return True
    try:
        if abs(float(pred_norm) - float(gt_norm)) < 0.0001:
            return True
    except (ValueError, OverflowError):
        pass
    pred_frac = _try_eval_frac(pred_norm)
    gt_frac = _try_eval_frac(gt_norm)
    try:
        pv = pred_frac if pred_frac is not None else float(pred_norm)
        gv = gt_frac if gt_frac is not None else float(gt_norm)
        if abs(pv - gv) < 0.0001:
            return True
    except (ValueError, TypeError):
        pass
    try:
        from sympy import simplify, S
        from sympy.parsing.latex import parse_latex
        import signal as _signal

        def _to_sym(s):
            if not s:
                return None
            for parser_name in ('latex', 'sympify'):
                try:
                    if parser_name == 'latex':
                        return parse_latex(s)
                    else:
                        from sympy import sympify as _sympify
                        return _sympify(s)
                except Exception:
                    continue
            return None

        def _handler(signum, frame):
            raise TimeoutError('sympy timeout')
        old = _signal.signal(_signal.SIGALRM, _handler)
        _signal.setitimer(_signal.ITIMER_REAL, 1.5)
        try:
            a = _to_sym(predicted.strip())
            b = _to_sym(ground_truth.strip())
            if a is not None and b is not None:
                if simplify(a - b) == 0 or simplify(a - b) == S.Zero:
                    _signal.setitimer(_signal.ITIMER_REAL, 0)
                    _signal.signal(_signal.SIGALRM, old)
                    return True
        except Exception:
            pass
        finally:
            _signal.setitimer(_signal.ITIMER_REAL, 0)
            _signal.signal(_signal.SIGALRM, old)
    except ImportError:
        pass
    pred_set = _normalize_set_answer(pred_norm)
    gt_set = _normalize_set_answer(gt_norm)
    if pred_set and gt_set and (pred_set == gt_set):
        return True
    return False

def load_math500(cache_dir: Optional[str]=None) -> list:
    ds = load_dataset('HuggingFaceH4/MATH-500', split='test', cache_dir=cache_dir)
    problems = []
    for i, item in enumerate(ds):
        answer = extract_boxed_answer(item.get('answer', item.get('solution', '')))
        problems.append(Problem(id=f'math500_{i}', question=item['problem'], answer=answer, raw_answer=item.get('answer', item.get('solution', '')), difficulty=str(item.get('level', '')), subject=item.get('type', item.get('subject', '')), source='MATH500'))
    return problems

def load_aime2024(data_path: Optional[str]=None, cache_dir: Optional[str]=None) -> list:
    if data_path and Path(data_path).exists():
        with open(data_path) as f:
            data = json.load(f)
        problems = []
        for i, item in enumerate(data):
            problems.append(Problem(id=f'aime2024_{i}', question=item['problem'], answer=str(item['answer']), raw_answer=item.get('solution', ''), difficulty='competition', source='AIME2024'))
        return problems
    try:
        ds = load_dataset('AI-MO/aimo-validation-aime', split='train', cache_dir=cache_dir)
        problems = []
        for i, item in enumerate(ds):
            problems.append(Problem(id=f'aime_{i}', question=item['problem'], answer=str(item.get('answer', '')), raw_answer=item.get('solution', ''), difficulty='competition', source='AIME'))
        return problems
    except Exception as e:
        print(f'Warning: Could not load AIME dataset: {e}')
        return []

def load_gpqa(cache_dir: Optional[str]=None) -> list:
    try:
        ds = load_dataset('Idavidrein/gpqa', 'gpqa_diamond', split='train', cache_dir=cache_dir)
    except Exception:
        try:
            ds = load_dataset('Idavidrein/gpqa', split='train', cache_dir=cache_dir)
        except Exception as e:
            print(f'Warning: GPQA requires HuggingFace authentication. Error: {e}')
            return []
    problems = []
    for i, item in enumerate(ds):
        question_text = item.get('Question', item.get('question', ''))
        choices = []
        for key in ['Correct Answer', 'Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3']:
            if key in item and item[key]:
                choices.append(item[key])
        if choices:
            import random
            random.seed(i)
            random.shuffle(choices)
            choice_labels = ['A', 'B', 'C', 'D']
            choice_text = '\n'.join((f'({label}) {choice}' for label, choice in zip(choice_labels, choices)))
            correct_idx = choices.index(item.get('Correct Answer', ''))
            correct_label = choice_labels[correct_idx]
            question_text = f'{question_text}\n\n{choice_text}'
            answer = correct_label
        else:
            answer = item.get('Correct Answer', '')
        problems.append(Problem(id=f'gpqa_{i}', question=question_text, answer=answer, raw_answer=item.get('Correct Answer', ''), difficulty='graduate', subject=item.get('Subdomain', item.get('subdomain', '')), source='GPQA'))
    return problems

def load_gsm8k(cache_dir: Optional[str]=None, split: str='test', limit: Optional[int]=None) -> list:
    ds = load_dataset('gsm8k', 'main', split=split, cache_dir=cache_dir)
    problems = []
    for i, item in enumerate(ds):
        if limit is not None and i >= limit:
            break
        raw_answer = item['answer']
        m = re.search('####\\s*([\\-\\d\\.,/]+)', raw_answer)
        answer = m.group(1).replace(',', '').strip() if m else raw_answer.strip()
        problems.append(Problem(id=f'gsm8k_{i}', question=item['question'], answer=answer, raw_answer=raw_answer, difficulty='grade-school', source='GSM8K'))
    return problems

def load_dataset_by_name(name: str, cache_dir: Optional[str]=None, **kwargs) -> list:
    loaders = {'math500': load_math500, 'aime2024': load_aime2024, 'aime': load_aime2024, 'gpqa': load_gpqa, 'gsm8k': load_gsm8k}
    loader = loaders.get(name.lower())
    if loader is None:
        raise ValueError(f'Unknown dataset: {name}. Available: {list(loaders.keys())}')
    return loader(cache_dir=cache_dir, **kwargs)
