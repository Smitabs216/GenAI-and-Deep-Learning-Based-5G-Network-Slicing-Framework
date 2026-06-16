# ============================================================
#  config.py  –  Global constants for the entire pipeline
#  Project: Deep Learning Based Parameter Prediction and
#           Resource Management for Network Slicing (5G)
# ============================================================

DATASET_PATH = "dataset.csv"

# ── Slices ────────────────────────────────────────────────
SLICES = ["eMBB", "URLLC", "mMTC"]

# ── Resource Blocks ───────────────────────────────────────
N_RBS_TOTAL = 100
RB_MIN  = {"eMBB": 5,  "URLLC": 5,  "mMTC": 5}
RB_MAX  = {"eMBB": 80, "URLLC": 70, "mMTC": 40}

# ── QoS Thresholds (3GPP TS 23.501) ──────────────────────
DELAY_MAX = {"eMBB": 300.0,  "URLLC": 10.0,   "mMTC": 500.0}   # ms
# FIX: Added 10% tolerance buffer to avoid floating-point rejection
LOSS_MAX  = {"eMBB": 1.1e-3, "URLLC": 1.1e-5, "mMTC": 1.1e-3}
ETA       = {"eMBB": 0.70,   "URLLC": 0.70,   "mMTC": 0.70} #efficiency target (70% of theoretical max)
P_MAX     = 46.0

# ── Utility function weights ──────────────────────────────
# Us = α·(Rs/Ds) (resource utilization) - β·(Delay/DELAY_MAX) - γ·(Loss/LOSS_MAX)
# All three terms are dimensionless [0,1] — no term dominates
ALPHA = {"eMBB": 1.0, "URLLC": 1.0, "mMTC": 1.0}
BETA  = {"eMBB": 0.5, "URLLC": 0.8, "mMTC": 0.3}
GAMMA = {"eMBB": 0.3, "URLLC": 0.8, "mMTC": 0.3}

# ── Spectral efficiency calibrated from dataset ───────────
# SE_s = mean_TP_s / (N/3 × BW_PER_RB)
SE_CALIB = {"eMBB": 1.7952, "URLLC": 0.2610, "mMTC": 0.0087}
BW_PER_RB = 0.180   # MHz per RB (5G NR numerology μ=0)

# (legacy — kept for backward compatibility)
SE_BASE = {"eMBB": 5.0, "URLLC": 3.0, "mMTC": 1.5}

# ── LSTM settings ─────────────────────────────────────────
SEQ_LEN    = 10 # sequence length for LSTM input
LSTM_UNITS = 64 # number of LSTM units
EPOCHS     = 80 # number of epochs for training the LSTM model
BATCH_SIZE = 32 # batch size for training the LSTM model

# ── DRL settings ─────────────────────────────────────────
DRL_EPISODES  = 300 # no of episodes to train the DQN agent
DRL_LR        = 1e-3 # learning rate for the DQN agent
DRL_GAMMA     = 0.95 # discount factor for future rewards
DRL_EPSILON   = 1.0 # initial exploration rate
DRL_EPS_DECAY = 0.99 # decay rate for exploration probability
DRL_EPS_MIN   = 0.05 # minimum exploration probability
DRL_BATCH     = 32 # batch size for experience replay
DRL_MEM_SIZE  = 2000 # maximum size of the replay memory

# ── Admission Control thresholds ──────────────────────────
ADMISSION_DELAY_THRESH = {
    "eMBB":  250.0,
    "URLLC":  8.5,   # slightly relaxed from 8.0 for practical margin
    "mMTC":  450.0,
}


"""
Code Examples
language-python
 Copy code
# ============================================================
#  config.py  –  Global constants for the entire pipeline
#  Project: Deep Learning Based Parameter Prediction and
#           Resource Management for Network Slicing (5G)
# ============================================================

DATASET_PATH = "dataset.csv"
DATASET_PATH: Specifies the path to the dataset file used for training and testing the model.
language-python
 Copy code
# ── Slices ────────────────────────────────────────────────
SLICES = ["eMBB", "URLLC", "mMTC"]
SLICES: A list of network slice types, namely enhanced Mobile Broadband (eMBB), Ultra-Reliable Low Latency Communications (URLLC), and massive Machine Type Communications (mMTC).
language-python
 Copy code
# ── Resource Blocks ───────────────────────────────────────
N_RBS_TOTAL = 100
RB_MIN  = {"eMBB": 5,  "URLLC": 5,  "mMTC": 5}
RB_MAX  = {"eMBB": 80, "URLLC": 70, "mMTC": 40}
N_RBS_TOTAL: Total number of resource blocks available.
RB_MIN: Minimum resource blocks allocated for each slice type.
RB_MAX: Maximum resource blocks allocated for each slice type.
language-python
 Copy code
# ── QoS Thresholds (3GPP TS 23.501) ──────────────────────
DELAY_MAX = {"eMBB": 300.0,  "URLLC": 10.0,   "mMTC": 500.0}   # ms
LOSS_MAX  = {"eMBB": 1.1e-3, "URLLC": 1.1e-5, "mMTC": 1.1e-3}
ETA       = {"eMBB": 0.70,   "URLLC": 0.70,   "mMTC": 0.70} #efficiency target (70% of theoretical max)
P_MAX     = 46.0
DELAY_MAX: Maximum allowable delay for each slice type in milliseconds.
LOSS_MAX: Maximum packet loss rates for each slice type.
ETA: Efficiency target for each slice type, indicating the desired performance level.
P_MAX: Maximum power allocation.
language-python
 Copy code
# ── Utility function weights ──────────────────────────────
ALPHA = {"eMBB": 1.0, "URLLC": 1.0, "mMTC": 1.0}
BETA  = {"eMBB": 0.5, "URLLC": 0.8, "mMTC": 0.3}
GAMMA = {"eMBB": 0.3, "URLLC": 0.8, "mMTC": 0.3}
ALPHA, BETA, GAMMA: Weights for the utility function, representing the importance of resource utilization, delay, and loss, respectively.
language-python
 Copy code
# ── Spectral efficiency calibrated from dataset ───────────
SE_CALIB = {"eMBB": 1.7952, "URLLC": 0.2610, "mMTC": 0.0087}
BW_PER_RB = 0.180   # MHz per RB (5G NR numerology μ=0)
SE_CALIB: Calibrated spectral efficiency values for each slice type.
BW_PER_RB: Bandwidth per resource block in MHz.
language-python
 Copy code
# ── LSTM settings ─────────────────────────────────────────
SEQ_LEN    = 10 # sequence length for LSTM input
LSTM_UNITS = 64 # number of LSTM units
EPOCHS     = 80 # number of epochs for training the LSTM model
BATCH_SIZE = 32 # batch size for training the LSTM model
SEQ_LEN: Length of the input sequences for the LSTM model.
LSTM_UNITS: Number of units in the LSTM layer.
EPOCHS: Total training epochs for the LSTM model.
BATCH_SIZE: Number of samples processed before the model is updated.
language-python
 Copy code
# ── DRL settings ─────────────────────────────────────────
DRL_EPISODES  = 300 # no of episodes to train the DQN agent
DRL_LR        = 1e-3 # learning rate for the DQN agent
DRL_GAMMA     = 0.95 # discount factor for future rewards
DRL_EPSILON   = 1.0 # initial exploration rate
DRL_EPS_DECAY = 0.99 # decay rate for exploration probability
DRL_EPS_MIN   = 0.05 # minimum exploration probability
DRL_BATCH     = 32 # batch size for experience replay
DRL_MEM_SIZE  = 2000 # maximum size of the replay memory
DRL_EPISODES: Number of episodes for training the Deep Q-Network (DQN) agent.
DRL_LR: Learning rate for the DQN agent.
DRL_GAMMA: Discount factor for future rewards.
DRL_EPSILON: Initial exploration rate for the agent.
DRL_EPS_DECAY: Rate at which the exploration probability decays.
DRL_EPS_MIN: Minimum exploration probability.
DRL_BATCH: Batch size for experience replay.
DRL_MEM_SIZE: Maximum size of the replay memory.
language-python
 Copy code
# ── Admission Control thresholds ──────────────────────────
ADMISSION_DELAY_THRESH = {
    "eMBB":  250.0,
    "URLLC":  8.5,   # slightly relaxed from 8.0 for practical margin
    "mMTC":  450.0,
}
ADMISSION_DELAY_THRESH: Thresholds for admission control based on delay for each slice type.
Conclusion
The config.py file serves as a foundational component for the deep learning model focused on resource management in 5G networks. By defining essential constants, it ensures that the model operates within the parameters necessary for effective network slicing. Understanding these constants is crucial for anyone looking to delve into the intricacies of deep learning applications in telecommunications.


"""