import json
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tts_benchmark.datasets.loader import check_answer

class PDE(nn.Module):

    def __init__(self, hidden_dim, mid_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(hidden_dim, mid_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mid_dim, mid_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(mid_dim // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)

def load_features(features_path):
    data = np.load(features_path, allow_pickle=True)
    return {'features': data['features'], 'pids': [str(p) for p in data['pids']], 'gt': [str(g) for g in data['gt']]}

def load_labels_from_sc(sc_results_path):
    with open(sc_results_path) as f:
        records = json.load(f)
    label_map = {}
    for r in records:
        pid = r['problem_id']
        gt = r['ground_truth']
        greedy_ans = r['samples'][0].get('answer', '')
        greedy_correct = check_answer(greedy_ans or '', gt)
        sc4_ans = r.get('final_answer', '')
        sc4_correct = check_answer(sc4_ans or '', gt)
        label_map[pid] = {'difficulty': 1 if not greedy_correct else 0, 'sc4_correct': sc4_correct, 'tokens_greedy': r['samples'][0].get('tokens', 0), 'consensus': r.get('consensus', 0)}
    return label_map

def train_one_fold(X_tr, y_tr, X_te, y_te, hidden_dim, n_epochs=30):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PDE(hidden_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.001)
    pos_w = (len(y_tr) - sum(y_tr)) / max(sum(y_tr), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w).to(device))
    Xt = torch.tensor(X_tr, dtype=torch.float32).to(device)
    yt = torch.tensor(y_tr, dtype=torch.float32).to(device)
    Xe = torch.tensor(X_te, dtype=torch.float32).to(device)
    best_auc = 0.0
    best_probs = None
    bs = 32
    for epoch in range(n_epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), bs):
            idx = perm[i:i + bs]
            logits = model(Xt[idx])
            loss = loss_fn(logits, yt[idx])
            optim.zero_grad()
            loss.backward()
            optim.step()
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(Xe)).cpu().numpy()
            try:
                auc = roc_auc_score(y_te, probs)
            except ValueError:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_probs = probs
    return (best_auc, best_probs, model)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='results/prefill_hidden_math500.npz')
    ap.add_argument('--labels', '--sc_results', dest='labels', default='results/sc4_baseline_math500.json')
    ap.add_argument('--out', default='results/pde_model.pt')
    ap.add_argument('--predictions_out', default=None)
    ap.add_argument('--n_folds', type=int, default=5)
    args = ap.parse_args()
    feats = load_features(args.features)
    labels = load_labels_from_sc(args.labels)
    aligned_X, aligned_y, aligned_pids = ([], [], [])
    for i, pid in enumerate(feats['pids']):
        if pid in labels:
            aligned_X.append(feats['features'][i])
            aligned_y.append(labels[pid]['difficulty'])
            aligned_pids.append(pid)
    X = np.stack(aligned_X)
    y = np.array(aligned_y, dtype=np.float32)
    print(f'Aligned: {len(y)} examples, hard rate (greedy wrong) = {y.mean():.3f}')
    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=42)
    aucs = []
    fold_probs = np.zeros(len(y))
    for fi, (tr, te) in enumerate(kf.split(X)):
        auc, probs, model = train_one_fold(X[tr], y[tr], X[te], y[te], hidden_dim=X.shape[1])
        aucs.append(auc)
        fold_probs[te] = probs
        print(f'Fold {fi + 1}: AUC = {auc:.4f}')
    print(f'\nMean AUC: {np.mean(aucs):.4f}  +/- {np.std(aucs):.4f}')
    bins = np.linspace(0, 1, 11)
    binned = np.digitize(fold_probs, bins) - 1
    print('\nCalibration:')
    for b in range(10):
        mask = binned == b
        if mask.sum() > 0:
            print(f'  bin [{bins[b]:.1f}, {bins[b + 1]:.1f}): n={mask.sum():3d} pred={fold_probs[mask].mean():.3f} actual_hard={y[mask].mean():.3f}')
    print('\nTraining final model on all data...')
    final_auc, _, final_model = train_one_fold(X, y, X, y, X.shape[1])
    torch.save({'state_dict': final_model.state_dict(), 'hidden_dim': X.shape[1], 'cv_auc_mean': float(np.mean(aucs)), 'cv_auc_std': float(np.std(aucs))}, args.out)
    print(f'Saved to {args.out}')
    pred_path = args.predictions_out or args.out.replace('.pt', '_predictions.json')
    preds = {}
    for i, pid in enumerate(aligned_pids):
        preds[pid] = {'predicted_difficulty': float(fold_probs[i]), 'label': int(labels[pid]['difficulty']), 'tokens_greedy': labels[pid]['tokens_greedy']}
    with open(pred_path, 'w') as f:
        json.dump(preds, f, indent=2)
    print(f'Saved predictions to {pred_path}')
if __name__ == '__main__':
    main()
