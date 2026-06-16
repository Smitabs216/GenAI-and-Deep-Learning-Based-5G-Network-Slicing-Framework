# ============================================================
#  resource_allocation.py
#  Three resource allocation methods:
#   1. Greedy   – priority-weighted proportional
#   2. Convex   – scipy SLSQP with QoS constraints
#   3. DRL(DQN) – PyTorch DQN agent
#
#  Objective:
#   max ∑_s Us   where
#   Us = α·(Rs/Ds) - β·(Delay/DELAY_MAX) - γ·(Loss/LOSS_MAX)
#
#  FIX 1: Throughput uses dataset-calibrated SE (not raw Shannon)
#  FIX 2: Utility penalty terms normalised → always in [0,1]
# ============================================================

import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from collections import deque
import random
from scipy.optimize import minimize, LinearConstraint, Bounds

from config import (SLICES, N_RBS_TOTAL, RB_MIN, RB_MAX,
                    ALPHA, BETA, GAMMA, ETA, DELAY_MAX, LOSS_MAX,
                    SE_CALIB, BW_PER_RB,
                    DRL_EPISODES, DRL_LR, DRL_GAMMA, DRL_EPSILON,
                    DRL_EPS_DECAY, DRL_EPS_MIN, DRL_BATCH, DRL_MEM_SIZE)

random.seed(42); np.random.seed(42); torch.manual_seed(42)


# ── Shared helpers ────────────────────────────────────────

def compute_throughput_from_rbs(rbs: dict, snr_db: dict = None) -> dict:
    """
    Rs = RBs × BW_PER_RB × SE_CALIB[s]   [Mbps]
    Uses dataset-calibrated spectral efficiency (realistic scale).
    snr_db kept as argument for API compatibility but not used.
    """
    return {slc: BW_PER_RB * rbs[slc] * SE_CALIB[slc] for slc in SLICES}


def utility(rbs: dict, demands: dict, snr_db: dict = None) -> dict:
    """
    Us = α·(Rs/Ds) - β·(Delay/DELAY_MAX) - γ·(Loss/LOSS_MAX)
    All penalty terms are normalised to [0, 1].
    """
    tp = compute_throughput_from_rbs(rbs)
    us = {}
    for slc in SLICES:
        Ds      = max(demands[slc]["Throughput_Mbps"], 1e-9)
        Delay_s = demands[slc]["Packet_Delay_Budget_ms"]
        Loss_s  = demands[slc]["Packet_Loss_Rate"]
        us[slc] = (ALPHA[slc] * (tp[slc] / Ds)
                   - BETA[slc]  * (Delay_s / DELAY_MAX[slc])   # normalised
                   - GAMMA[slc] * (Loss_s  / LOSS_MAX[slc]))    # normalised
    return us


def total_utility(rbs, demands, snr_db=None):
    return sum(utility(rbs, demands).values())


def check_qos_constraints(rbs: dict, demands: dict, snr_db: dict = None):
    tp = compute_throughput_from_rbs(rbs)
    violations = []
    for slc in SLICES:
        Ds    = max(demands[slc]["Throughput_Mbps"], 1e-9)
        delay = demands[slc]["Packet_Delay_Budget_ms"]
        loss  = demands[slc]["Packet_Loss_Rate"]
        if tp[slc] / Ds < ETA[slc]:
            violations.append(f"{slc}: demand ratio {tp[slc]/Ds:.2f} < {ETA[slc]}")
        if delay > DELAY_MAX[slc]:
            violations.append(f"{slc}: delay {delay:.1f} > {DELAY_MAX[slc]}")
        if loss > LOSS_MAX[slc]:
            violations.append(f"{slc}: PLR {loss:.2e} > {LOSS_MAX[slc]:.2e}")
    return violations


def default_snr(df, slc):
    return float(df[df["Slice_Type"] == slc]["SNR_dB"].mean())


# ══════════════════════════════════════════════════════════
# METHOD 1 — GREEDY
# ══════════════════════════════════════════════════════════

def greedy_allocation(demands: dict, snr_db: dict = None) -> dict:
    """Priority score = α / (β·(D/Dmax) + γ·(L/Lmax) + ε)
    Enforces demand-based minimum RBs before proportional split."""
    # Step 1: Compute demand-based minimums (satisfy ETA% of demand)
    rb_demand_min = {}
    for slc in SLICES:
        Ds       = max(demands[slc]["Throughput_Mbps"], 1e-9)
        min_tp   = ETA[slc] * Ds
        coeff    = BW_PER_RB * SE_CALIB[slc]
        rb_demand_min[slc] = max(RB_MIN[slc], int(np.ceil(min_tp / coeff)))

    # Step 2: Compute priority scores
    scores = {}
    for slc in SLICES:
        d_norm = demands[slc]["Packet_Delay_Budget_ms"] / DELAY_MAX[slc]
        l_norm = demands[slc]["Packet_Loss_Rate"]       / LOSS_MAX[slc]
        scores[slc] = ALPHA[slc] / (BETA[slc] * d_norm + GAMMA[slc] * l_norm + 1e-9)

    # Step 3: Remaining budget after demand minimums
    used_min = sum(rb_demand_min.values())
    budget   = max(0, N_RBS_TOTAL - used_min)
    total_score = sum(scores.values())

    rbs = {}
    for slc in SLICES:
        extra    = int((scores[slc] / total_score) * budget)
        rbs[slc] = max(rb_demand_min[slc],
                       min(RB_MAX[slc], rb_demand_min[slc] + extra))

    # Step 4: Trim if over budget
    total_used = sum(rbs.values())
    if total_used > N_RBS_TOTAL:
        excess = total_used - N_RBS_TOTAL
        for slc in sorted(SLICES, key=lambda s: scores[s]):
            cut = min(excess, rbs[slc] - rb_demand_min[slc])
            rbs[slc] -= cut; excess -= cut
            if excess == 0: break

    print(f"  [Greedy]  RBs: {rbs}  total={sum(rbs.values())}")
    return rbs


# ══════════════════════════════════════════════════════════
# METHOD 2 — CONVEX OPTIMISATION
# ══════════════════════════════════════════════════════════

def convex_allocation(demands: dict, snr_db: dict = None) -> dict:
    def neg_u(x):
        rbs = {s: x[i] for i, s in enumerate(SLICES)}
        return -total_utility(rbs, demands)

    def neg_u_grad(x):
        eps  = 1e-4; f0 = neg_u(x)
        grad = np.zeros_like(x)
        for j in range(len(x)):
            xp = x.copy(); xp[j] += eps
            grad[j] = (neg_u(xp) - f0) / eps
        return grad

    x0 = np.array([20.0, 40.0, 40.0])
    lb  = np.array([RB_MIN[s] for s in SLICES], dtype=float)
    ub  = np.array([RB_MAX[s] for s in SLICES], dtype=float)

    # Total RB constraint
    lc   = LinearConstraint(np.ones((1, 3)), lb=0, ub=N_RBS_TOTAL)
    cons = [lc]
    # Demand satisfaction per slice
    for i, slc in enumerate(SLICES):
        Ds    = max(demands[slc]["Throughput_Mbps"], 1e-9)
        coeff = BW_PER_RB * SE_CALIB[slc]          # Rs_i = coeff * x_i
        A = np.zeros((1, 3)); A[0, i] = coeff
        cons.append(LinearConstraint(A, lb=ETA[slc] * Ds, ub=np.inf))

    result = minimize(neg_u, x0, jac=neg_u_grad, method="SLSQP",
                      bounds=Bounds(lb, ub), constraints=cons,
                      options={"maxiter": 500, "ftol": 1e-8})

    rbs = {}
    for i, slc in enumerate(SLICES):
        rbs[slc] = int(np.clip(round(result.x[i]), RB_MIN[slc], RB_MAX[slc]))
    while sum(rbs.values()) > N_RBS_TOTAL:
        slc_max = max(rbs, key=lambda s: rbs[s] - RB_MIN[s])
        rbs[slc_max] -= 1

    print(f"  [Convex]  RBs: {rbs}  total={sum(rbs.values())}  "
          f"converged={result.success}")
    return rbs


# ══════════════════════════════════════════════════════════
# METHOD 3 — DEEP REINFORCEMENT LEARNING (DQN)
# ══════════════════════════════════════════════════════════

class SlicingEnv:
    def __init__(self, demands, snr_db=None, n_action_steps=10):
        self.demands   = demands
        self.state_dim = 9
        self._build_action_space(n_action_steps)

    def _build_action_space(self, steps):
        actions = []
        step = max(1, (N_RBS_TOTAL - sum(RB_MIN.values())) // steps)
        for e in range(RB_MIN["eMBB"],  RB_MAX["eMBB"]  + 1, step):
            for u in range(RB_MIN["URLLC"], RB_MAX["URLLC"] + 1, step):
                m = N_RBS_TOTAL - e - u
                if RB_MIN["mMTC"] <= m <= RB_MAX["mMTC"]:
                    actions.append({"eMBB": e, "URLLC": u, "mMTC": m})
        self.action_space = actions if actions else [{"eMBB": 40, "URLLC": 40, "mMTC": 20}]
        self.n_actions    = len(self.action_space)

    def _get_state(self):
        s = []
        for slc in SLICES:
            s += [self.demands[slc]["Throughput_Mbps"],
                  self.demands[slc]["Packet_Delay_Budget_ms"] / DELAY_MAX[slc],
                  self.demands[slc]["Packet_Loss_Rate"]       / LOSS_MAX[slc]]
        return np.array(s, dtype=np.float32)

    def step(self, action_idx):
        rbs    = self.action_space[action_idx]
        reward = total_utility(rbs, self.demands)
        viols  = check_qos_constraints(rbs, self.demands)
        reward -= 5.0 * len(viols)
        return self._get_state(), reward, True

    def reset(self):
        return self._get_state()


class DQNNet(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128),        nn.ReLU(),
            nn.Linear(128, 64),         nn.ReLU(),
            nn.Linear(64, n_actions),
        )
    def forward(self, x): return self.net(x)


class DQNAgent:
    def __init__(self, state_dim, n_actions):
        self.n_actions  = n_actions
        self.epsilon    = DRL_EPSILON
        self.gamma      = DRL_GAMMA
        self.memory     = deque(maxlen=DRL_MEM_SIZE)
        self.policy_net = DQNNet(state_dim, n_actions)
        self.target_net = DQNNet(state_dim, n_actions)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=DRL_LR)
        self.loss_fn    = nn.MSELoss()

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        with torch.no_grad():
            q = self.policy_net(torch.FloatTensor(state))
        return int(q.argmax())

    def remember(self, s, a, r, s2, done):
        self.memory.append((s, a, r, s2, done))

    def learn(self):
        if len(self.memory) < DRL_BATCH: return
        batch = random.sample(self.memory, DRL_BATCH)
        s,a,r,s2,d = zip(*batch)
        S  = torch.FloatTensor(np.array(s))
        A  = torch.LongTensor(a)
        R  = torch.FloatTensor(r)
        S2 = torch.FloatTensor(np.array(s2))
        D  = torch.FloatTensor(d)
        q_vals   = self.policy_net(S).gather(1, A.unsqueeze(1)).squeeze()
        q_next   = self.target_net(S2).max(1)[0].detach()
        q_target = R + self.gamma * q_next * (1 - D)
        loss = self.loss_fn(q_vals, q_target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        if self.epsilon > DRL_EPS_MIN:
            self.epsilon *= DRL_EPS_DECAY

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())


def drl_allocation(demands: dict, snr_db: dict = None,
                   episodes: int = DRL_EPISODES) -> tuple:
    env   = SlicingEnv(demands)
    agent = DQNAgent(env.state_dim, env.n_actions)
    best_reward = -np.inf
    best_rbs    = env.action_space[0]
    reward_hist = []
    for ep in range(episodes):
        state  = env.reset()
        action = agent.act(state)
        ns, reward, done = env.step(action)
        agent.remember(state, action, reward, ns, float(done))
        agent.learn()
        if ep % 20 == 0: agent.update_target()
        reward_hist.append(reward)
        if reward > best_reward:
            best_reward = reward; best_rbs = env.action_space[action]
    print(f"  [DRL]     RBs: {best_rbs}  total={sum(best_rbs.values())}  "
          f"best_reward={best_reward:.4f}  episodes={episodes}")
    return best_rbs, reward_hist


def run_all_methods(demands: dict, df) -> tuple:
    snr_db = {slc: default_snr(df, slc) for slc in SLICES}
    print("\n── Resource Allocation ──")
    rbs_greedy           = greedy_allocation(demands, snr_db)
    rbs_convex           = convex_allocation(demands, snr_db)
    rbs_drl, drl_rewards = drl_allocation(demands, snr_db)
    results = {}
    for name, rbs in [("Greedy", rbs_greedy),
                       ("Convex", rbs_convex),
                       ("DRL",    rbs_drl)]:
        us    = utility(rbs, demands)
        tp    = compute_throughput_from_rbs(rbs)
        viols = check_qos_constraints(rbs, demands)
        results[name] = {"rbs": rbs, "utility": us,
                         "total_utility": sum(us.values()),
                         "throughput": tp, "violations": viols}
    return results, snr_db, drl_rewards
