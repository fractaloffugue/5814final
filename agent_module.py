# =========================
# 0. Import libraries
# =========================
import pandas as pd
import numpy as np
import random
from copy import deepcopy
from scipy.stats import entropy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, ParameterGrid
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# =========================
# 0.5 Hyperparameters & thresholds
# =========================
DENSITY_QUANTILE    = 50      
MARGIN_QUANTILE     = 15      
RL_CONF_THRESHOLD   = 0.6     
EWMA_MEAN           = 0.5     
EXP4P_ROUNDS        = 10      
EXPLORATION_RATE    = 0.15    
EPOCHS              = 100      
PATIENCE            = 10      
CV_FOLDS            = 3       
HYPERPARAMS         = {       
    'hidden_dims': [ (64, 32)],
    'lr': [1e-3]
}

# =========================
# 1. Define Agent Classes
# =========================
class ALAgent_X_MMd_B:
    """Max–Min distance AL agent."""
    def __init__(self, DIM, window_size, density_threshold, budget=None):
        self._q = DIM
        self._Size = window_size
        self._d_threshod = density_threshold
        self._budget = budget
        self._n_seen = 0
        self._n_acquired_samples = 0
        self._W = []
        self._probs = []

    def NNlocalDensity(self, x_t):
        if self._n_seen < 2:
            self._W.append(x_t)
            return True
        W_array = np.vstack(self._W)
        D_mat = euclidean_distances(W_array)
        np.fill_diagonal(D_mat, np.nan)
        min_dists = np.nanmin(D_mat, axis=1)
        u = x_t.flatten()
        dists = np.linalg.norm(W_array - u, axis=1)
        prob = np.min(dists) / np.max(min_dists)
        prob = min(prob, 1.0)
        self._probs.append(prob)
        decision = bool(np.random.binomial(1, prob))
        # update sliding window
        if len(self._W) < self._Size:
            self._W.append(x_t)
        else:
            self._W.pop(0)
            self._W.append(x_t)
        return decision

    def get_agent_decision(self, x_t, certainty=None, margin=None):
        dec = self.NNlocalDensity(x_t.reshape(1, -1))
        self._n_seen += 1
        if dec:
            self._n_acquired_samples += 1
        return [1.0 - float(dec), float(dec)]


class ALAgent_X_hD_B:
    """High‑Dimensional density AL agent."""
    def __init__(self, DIM, window_size, density_threshold, budget=None):
        self._q = DIM
        self._Size = window_size
        self._d_threshod = density_threshold
        self._budget = budget
        self._n_seen = 0
        self._n_acquired_samples = 0
        self._W = []

    def NNlocalDensity(self, x_t):
        if self._n_seen < 2:
            self._W.append(x_t)
            return True
        W_array = np.vstack(self._W)
        D_mat = euclidean_distances(W_array)
        np.fill_diagonal(D_mat, np.nan)
        max_dists = np.nanmax(D_mat, axis=1)
        u = x_t.flatten()
        dists = np.linalg.norm(W_array - u, axis=1)
        count = np.sum(dists > max_dists * self._d_threshod)
        prob = min(count / (len(self._W) * self._d_threshod), 1.0)
        decision = bool(np.random.binomial(1, prob))
        if len(self._W) < self._Size:
            self._W.append(x_t)
        else:
            self._W.pop(0)
            self._W.append(x_t)
        return decision

    def get_agent_decision(self, x_t, certainty=None, margin=None):
        dec = self.NNlocalDensity(x_t.reshape(1, -1))
        self._n_seen += 1
        if dec:
            self._n_acquired_samples += 1
        return [1.0 - float(dec), float(dec)]


class ALAgent_X_M_B:
    """Margin‑based AL agent."""
    def __init__(self, adjust_s, window_size, threshold_margin, budget=None):
        self._s = adjust_s
        self._Size = window_size
        self._threshold_margin = threshold_margin
        self._budget = budget
        self._n_seen = 0
        self._n_acquired_samples = 0
        self._W = []

    def NNlocalDensity(self, x_t):
        if self._n_seen < 2:
            self._W.append(x_t)
            return True
        W_array = np.vstack(self._W)
        D_mat = euclidean_distances(W_array)
        np.fill_diagonal(D_mat, np.nan)
        min_dists = np.nanmin(D_mat, axis=1)
        u = x_t.flatten()
        dists = np.linalg.norm(W_array - u, axis=1)
        count = np.sum(dists < min_dists)
        return bool(count > 0)

    def get_agent_decision(self, x_t, certainty, margin):
        self._n_seen += 1
        dec = False
        if self.NNlocalDensity(x_t):
            random_margin = np.random.normal(0, 1) * self._threshold_margin
            dec = (margin < random_margin)
            if dec:
                self._threshold_margin *= (1 - self._s)
                self._n_acquired_samples += 1
            else:
                self._threshold_margin *= (1 + self._s)
        if len(self._W) < self._Size:
            self._W.append(x_t)
        else:
            self._W.pop(0)
            self._W.append(x_t)
        return [1.0 - float(dec), float(dec)]


class RAL_B:
    """Basic RAL with uncertainty + ε‑greedy."""
    def __init__(self, threshold_uncertainty, threshold_greedy, eta, budget=None):
        self._threshold_uncertainty = threshold_uncertainty
        self._threshold_greedy = threshold_greedy
        self._eta = eta
        self._alpha = [1.0]
        self._n_seen = 0
        self._n_acquired_samples = 0
        self.label_decision = False

    def ask_committee(self, x_certainty):
        committee_decision = (x_certainty < self._threshold_uncertainty)
        return committee_decision, [committee_decision]

    def should_label(self, committee_decision):
        self._n_seen += 1
        if random.random() < self._threshold_greedy:
            self.label_decision = True
        else:
            self.label_decision = bool(committee_decision)
        if self.label_decision:
            self._n_acquired_samples += 1

    def get_agent_decision(self, x_t, certainty, margin=None):
        comm, _ = self.ask_committee(certainty)
        self.should_label(comm)
        return [1.0 - float(self.label_decision), float(self.label_decision)]


class RAL_B_EXP4P(RAL_B):
    """RAL with EXP4.P weight updates."""
    def __init__(self, threshold_uncertainty, eta, reward, penalty, mode):
        super().__init__(threshold_uncertainty, 0.0, eta)
        self._reward = reward
        self._penalty = penalty
        self._mode = mode
    


class Exp4P_EN_SWAP:
    """EXP4.P with EWMA and swap mechanism."""
    def __init__(self, committee_agents, max_T, delta, p_min, reward, penalty,
                 budget=None, budget_mode='soft', greedy_threshold=0.01,
                 mode='ewma', ewma_k=5, ewma_Lambda=0.3, ewma_mu=0.5):
        self._committee_agents   = committee_agents
        self._K                  = 2
        self._N                  = len(committee_agents)
        self._delta              = delta
        self._reward             = reward
        self._penalty            = penalty
        self._budget_mode        = budget_mode
        self._threshold_greedy   = greedy_threshold
        self._p_min              = p_min or np.sqrt(np.log(self._N)/(self._K*max_T))
        w0 = np.ones(self._N)
        self._model = {
            'w':     [w0.tolist()],
            'w_std': [(w0 / w0.sum()).tolist()]
        }
        self._all_decision = []
        self._exp_decision = []
        self.label_decision = False
        self._mode          = mode
        self._ewma_k        = ewma_k
        self._ewma_Lambda   = ewma_Lambda
        self._ewma_mu       = ewma_mu
        self._E_t           = []

    
    def ask_committee(self, x_t, certainty, margin):
        decs = [
            agent.get_agent_decision(x_t, certainty, margin)[1]
            for agent in self._committee_agents
        ]
        self._all_decision.append(decs)
        w_std = np.array(self._model['w_std'][-1])
        probs = []
        for action in [0, 1]:
            weighted = [
                w_std[i] * ((1 - action) + action * decs[i])
                for i in range(self._N)
            ]
            p = (1 - self._K * self._p_min) * sum(weighted) + self._p_min
            probs.append(p)
        probs = np.array(probs)
        probs /= probs.sum()
        self._exp_decision.append(probs.tolist())
        if random.random() < self._threshold_greedy:
            choice = 1
        else:
            choice = int(np.argmax(probs))
        self.label_decision = bool(choice)
        return self.label_decision, probs.tolist(), decs

    
    def get_agent_decision(self, x_t, certainty, margin=None):
        dec, _, _ = self.ask_committee(x_t, certainty, margin)
        return [1.0 - float(dec), float(dec)]


# =========================
# 2. Data Preprocessing (spambase dataset)
# =========================
column_names = [f'feature_{i}' for i in range(57)] + ['label']
df = pd.read_csv(r"D:\zhuomian\10 paper\spambase\spambase.data",
                 header=None, names=column_names)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Downsample majority class to balance
Xy = np.hstack((X_train, y_train.reshape(-1,1)))
majority = Xy[y_train == 0]
minority = Xy[y_train == 1]
maj_down = resample(majority, replace=False, n_samples=len(minority), random_state=42)
balanced = np.vstack((minority, maj_down))
np.random.shuffle(balanced)
X_train_balanced = balanced[:, :-1]
y_train_balanced = balanced[:, -1]

# Prepare test loader
X_test_tensor  = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor  = torch.tensor(y_test, dtype=torch.float32)
test_dataset    = TensorDataset(X_test_tensor, y_test_tensor)
test_loader     = DataLoader(test_dataset, batch_size=64, shuffle=False)


# =========================
# 3. Define MLP Classifier
# =========================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128,64)):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)


# =========================
# 4. Instantiate Agents
# =========================
input_dim      = X_train_balanced.shape[1]
window_size    = 50
density_thresh = DENSITY_QUANTILE / 100.0
margin_thresh  = MARGIN_QUANTILE / 100.0
adjust_s       = 0.1

al1 = ALAgent_X_MMd_B(input_dim, window_size, density_thresh)
al2 = ALAgent_X_hD_B(input_dim, window_size, density_thresh)
al3 = ALAgent_X_M_B(adjust_s, window_size, margin_thresh)

rl1 = RAL_B(RL_CONF_THRESHOLD, EXPLORATION_RATE, eta=1.0)
rl2 = RAL_B_EXP4P(RL_CONF_THRESHOLD, eta=1.0, reward=1, penalty=-1, mode='const')
rl3 = Exp4P_EN_SWAP(
    committee_agents=[al1, al2, al3],
    max_T=X_train_balanced.shape[0],
    delta=0.1,
    p_min=None,
    reward=1,
    penalty=-1,
    greedy_threshold=EXPLORATION_RATE,
    ewma_mu=EWMA_MEAN
)

agents = [al1, al2, al3, rl1, rl2, rl3]


# =========================
# 5. Helper to collect agent decisions
# =========================
def get_agent_outputs(X, model, agents):
    probs   = torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).detach().numpy().flatten()
    margins = np.abs(probs - 0.5)
    outputs = []
    for agent in agents:
        decisions = []
        for x, p, m in zip(X, probs, margins):
            dec = agent.get_agent_decision(x.reshape(1,-1), p, m)
            if isinstance(dec, (list, np.ndarray)):
                dec = int(dec[1] >= 0.5)
            decisions.append(dec)
        outputs.append(np.array(decisions))
    return outputs


# =========================
# 6. EXP4.P‑EWMA Fusion & Reward
# =========================
def exp4p_fusion(agent_outputs, reward_fn, y_true, y_pred):
    N         = len(agent_outputs)
    alpha     = np.ones(N)
    decisions = np.vstack(agent_outputs)
    for _ in range(EXP4P_ROUNDS):
        explore_mask = (np.random.rand(*decisions.shape) < EXPLORATION_RATE)
        decisions_r  = np.where(explore_mask, 1 - decisions, decisions)
        fused_r      = (alpha @ decisions_r / alpha.sum()) >= 0.5
        rewards      = reward_fn(y_true, y_pred, fused_r)
        for i in range(N):
            alpha[i] *= np.exp(rewards * decisions_r[i])
        alpha /= alpha.sum()
    final = (alpha @ decisions / alpha.sum()) >= 0.5
    return final.astype(int)

def reward_fn(y_true, y_pred, selected):
    wrong = (y_true != y_pred)
    return (wrong & selected)*2.0 + (~wrong & selected)*(-1.0)



# =========================
# 7. Training utilities
# =========================
def train_model(model, train_loader, val_loader, lr, epochs, patience):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    best_loss = float('inf')
    wait      = 0
    best_w    = None
    for ep in range(epochs):
        model.train()
        for Xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(Xb).squeeze(), yb)
            loss.backward(); optimizer.step()
        model.eval()
        losses = []
        with torch.no_grad():
            for Xv, yv in val_loader:
                losses.append(criterion(model(Xv).squeeze(), yv).item())
        val_loss = np.mean(losses)
        if val_loss + 1e-4 < best_loss:
            best_loss, wait, best_w = val_loss, 0, model.state_dict()
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_w)
    return model

def hyperparameter_search(X, y):
    skf = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    best_score, best_params = -1, None
    for params in ParameterGrid(HYPERPARAMS):
        scores = []
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                                  torch.tensor(y_tr, dtype=torch.float32))
            va_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                                  torch.tensor(y_va, dtype=torch.float32))
            tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True)
            va_ld = DataLoader(va_ds, batch_size=64, shuffle=False)
            mdl = MLPClassifier(X.shape[1], hidden_dims=params['hidden_dims'])
            mdl = train_model(mdl, tr_ld, va_ld, params['lr'], EPOCHS, PATIENCE)
            preds, labs = [], []
            mdl.eval()
            with torch.no_grad():
                for Xb, yb in va_ld:
                    out = (mdl(Xb).squeeze().numpy() >= 0.5).astype(int)
                    preds.extend(out.tolist()); labs.extend(yb.numpy().astype(int).tolist())
            scores.append(f1_score(labs, preds))
        avg = np.mean(scores)
        if avg > best_score:
            best_score, best_params = avg, params
    return best_params

def train_and_eval(X_tr_full, y_tr_full):
    best_p = hyperparameter_search(X_tr_full, y_tr_full)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_tr_full, y_tr_full, test_size=0.2, random_state=42, stratify=y_tr_full
    )
    tr_ds = TensorDataset(torch.tensor(X_tr, dtype=torch.float32),
                          torch.tensor(y_tr, dtype=torch.float32))
    va_ds = TensorDataset(torch.tensor(X_va, dtype=torch.float32),
                          torch.tensor(y_va, dtype=torch.float32))
    tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=64, shuffle=False)
    mdl = MLPClassifier(X_tr_full.shape[1], hidden_dims=best_p['hidden_dims'])
    mdl = train_model(mdl, tr_ld, va_ld, best_p['lr'], EPOCHS, PATIENCE)
    preds, labs = [], []
    mdl.eval()
    with torch.no_grad():
        for Xb, yb in test_loader:
            out = (mdl(Xb).squeeze().numpy() >= 0.5).astype(int)
            preds.extend(out.tolist()); labs.extend(yb.numpy().astype(int).tolist())
    return accuracy_score(labs, preds), f1_score(labs, preds)


# =========================
# 8. Run Experiments
# =========================
initial_model = MLPClassifier(input_dim)
_ = train_and_eval(X_train_balanced, y_train_balanced)

agent_outputs = get_agent_outputs(X_train_balanced, initial_model, agents)

results = {}
for k in [2, 4, 6]:
    idx      = list(range(k))
    selected = np.any([agent_outputs[i] for i in idx], axis=0)
    X_sub    = X_train_balanced[selected]
    y_sub    = y_train_balanced[selected]
    acc, f1  = train_and_eval(X_sub, y_sub)
    results[f"Agent_{k}"] = (len(X_sub), acc, f1)

# Baseline on full balanced set
acc, f1 = train_and_eval(X_train_balanced, y_train_balanced)
results['Baseline'] = (len(X_train_balanced), acc, f1)

for name, (n, acc, f1) in results.items():
    print(f"{name}: Samples={n}, Accuracy={acc:.4f}, F1={f1:.4f}")
