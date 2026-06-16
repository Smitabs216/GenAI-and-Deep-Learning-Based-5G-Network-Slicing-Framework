"""
Regenerate the dataset with temporal correlation so LSTM can
actually learn patterns. Uses sinusoidal load cycles + AR(1)
noise — mimics real network traffic (rush hours, off-peak, etc.)
"""
import numpy as np
import pandas as pd

np.random.seed(42)

SLICES = ["eMBB", "URLLC", "mMTC"]
N = 600   # total rows per slice → 1800 total

# 3GPP base values per slice
BASE = {
    "eMBB":  {"tp": 10.0,  "delay": 150.0, "plr": 5e-4},
    "URLLC": {"tp": 1.5,   "delay": 7.0,   "plr": 1e-5},
    "mMTC":  {"tp": 0.05,  "delay": 300.0, "plr": 1e-4},
}
SLICE_TYPE_MAP = {
    "eMBB":  {"qci": [1,2,3,4,5,6,7,8,9],   "rtype":"Non-GBR",       "bw":[50,100,200], "mod":"64-QAM",  "pri": 50},
    "URLLC": {"qci": [80,82,83,84,85,86],    "rtype":"Delay-Crit GBR","bw":[10,20,50],   "mod":"16-QAM",  "pri": 20},
    "mMTC":  {"qci": [87,88,89,90,91,92,93], "rtype":"Non-GBR",       "bw":[1,5,10],     "mod":"BPSK",    "pri": 60},
}
DEVICES = {
    "eMBB":  ["Smartphone","Tablet","AR Headset","Laptop","Smart TV"],
    "URLLC": ["Autonomous Vehicle","Industrial Robot","Drone","V2X OBU","Remote Surgery Unit"],
    "mMTC":  ["IoT Sensor","Smart Meter","NB-IoT Module","Asset Tracker","Environmental Probe"],
}
USE_CASES = {
    "eMBB":  ["Live Streaming","VR Gaming","Video Conferencing","Cloud Download","AR Navigation"],
    "URLLC": ["V2X Control","Remote Surgery","Industrial Automation","Drone Navigation","Smart Grid"],
    "mMTC":  ["Smart Metering","Asset Tracking","Environmental Monitoring","Fleet Management","Smart Parking"],
}

rows = []
row_id = 1

for slc in SLICES:
    b   = BASE[slc]
    cfg = SLICE_TYPE_MAP[slc]

    # ── Time axis (simulate 10 hours, one sample per minute) ──
    t = np.linspace(0, 10 * np.pi, N)

    # ── Network load: sinusoidal rush-hour pattern ──
    load = 50 + 35 * np.sin(t / 3) + 10 * np.sin(t / 1.2)
    load = np.clip(load + np.random.normal(0, 3, N), 10, 99)

    # ── Throughput: AR(1) process tied to load ──
    tp = np.zeros(N)
    tp[0] = b["tp"]
    for i in range(1, N):
        tp[i] = 0.85 * tp[i-1] + 0.15 * b["tp"] * (1 + 0.4 * np.sin(t[i]/4)) \
                + np.random.normal(0, b["tp"] * 0.05)
    tp = np.clip(tp, b["tp"] * 0.3, b["tp"] * 2.5)

    # ── Delay: inversely correlated with load (more load → more delay) ──
    delay = np.zeros(N)
    delay[0] = b["delay"]
    for i in range(1, N):
        delay[i] = 0.80 * delay[i-1] + 0.20 * b["delay"] * (1 + 0.3 * (load[i]/100)) \
                   + np.random.normal(0, b["delay"] * 0.04)
    delay = np.clip(delay, b["delay"] * 0.5, b["delay"] * 1.8)

    # ── PLR: log-normal AR(1), spikes during high load ──
    log_plr = np.zeros(N)
    log_plr[0] = np.log10(b["plr"])
    for i in range(1, N):
        spike = 0.5 if load[i] > 85 else 0.0
        log_plr[i] = 0.88 * log_plr[i-1] + 0.12 * np.log10(b["plr"]) \
                     + spike + np.random.normal(0, 0.08)
    plr = np.clip(10 ** log_plr, 1e-8, 0.1)

    # ── Correlated secondary features ──
    snr    = 30 - 10 * (load / 100) + np.random.normal(0, 1.5, N)
    snr    = np.clip(snr, 5, 35)
    jitter = delay * np.random.uniform(0.03, 0.10, N)
    sig    = -60 - 20 * (load / 100) + np.random.normal(0, 2, N)
    sig    = np.clip(sig, -95, -55)
    num_ues = (load / 100 * (500 if slc == "mMTC" else 80)).astype(int) + \
              np.random.randint(1, 5, N)
    bw_arr  = np.random.choice(cfg["bw"], N)
    gbr_arr = tp * np.where(cfg["rtype"] != "Non-GBR",
                             np.random.uniform(0.5, 0.9, N), 0.0)

    for i in range(N):
        rows.append({
            "Row_ID":                   row_id,
            "Slice_Type":               slc,
            "5QI_Class":                int(np.random.choice(cfg["qci"])),
            "Resource_Type":            cfg["rtype"],
            "QoS_Priority":             cfg["pri"] + np.random.randint(-5, 5),
            "Use_Case":                 np.random.choice(USE_CASES[slc]),
            "Device_Type":              np.random.choice(DEVICES[slc]),
            "Packet_Delay_Budget_ms":   round(float(delay[i]), 4),
            "Packet_Loss_Rate":         float(plr[i]),
            "Throughput_Mbps":          round(float(tp[i]), 6),
            "Guaranteed_Bit_Rate_Mbps": round(float(gbr_arr[i]), 6),
            "Bandwidth_MHz":            int(bw_arr[i]),
            "Signal_Strength_dBm":      round(float(sig[i]), 2),
            "SNR_dB":                   round(float(snr[i]), 2),
            "Jitter_ms":                round(float(jitter[i]), 4),
            "Network_Load_Percent":     round(float(load[i]), 2),
            "Duration_s":               round(float(np.random.uniform(0.5, 120)), 2),
            "Modulation_Type":          cfg["mod"],
            "Num_UEs":                  int(num_ues[i]),
        })
        row_id += 1

df = pd.DataFrame(rows)
df.to_csv("dataset.csv", index=False)
print(f"Dataset: {df.shape}")
print(df.groupby("Slice_Type").size())
print(f"Distinct PDB : {df['Packet_Delay_Budget_ms'].nunique()}")
print(f"Distinct PLR : {df['Packet_Loss_Rate'].nunique()}")
print(f"Distinct TP  : {df['Throughput_Mbps'].nunique()}")



"""
Here is a detailed breakdown of the code provided:

language-python
 Copy code
import numpy as np
import pandas as pd
Imports: The code begins by importing the NumPy library for numerical operations and the Pandas library for data manipulation.
language-python
 Copy code
np.random.seed(42)
Random Seed: Sets the random seed for reproducibility, ensuring that the same random numbers are generated each time the code is run.
language-python
 Copy code
SLICES = ["eMBB", "URLLC", "mMTC"]
N = 600   # total rows per slice → 1800 total
Slice Types and Rows: Defines the three types of network slices and sets the number of rows to generate for each slice.
language-python
 Copy code
BASE = {
    "eMBB":  {"tp": 10.0,  "delay": 150.0, "plr": 5e-4},
    "URLLC": {"tp": 1.5,   "delay": 7.0,   "plr": 1e-5},
    "mMTC":  {"tp": 0.05,  "delay": 300.0, "plr": 1e-4},
}
Base Values: A dictionary containing base throughput (tp), delay, and packet loss rate (plr) for each slice type.
language-python
 Copy code
SLICE_TYPE_MAP = {
    "eMBB":  {"qci": [1,2,3,4,5,6,7,8,9],   "rtype":"Non-GBR",       "bw":[50,100,200], "mod":"64-QAM",  "pri": 50},
    "URLLC": {"qci": [80,82,83,84,85,86],    "rtype":"Delay-Crit GBR","bw":[10,20,50],   "mod":"16-QAM",  "pri": 20},
    "mMTC":  {"qci": [87,88,89,90,91,92,93], "rtype":"Non-GBR",       "bw":[1,5,10],     "mod":"BPSK",    "pri": 60},
}
Slice Configuration: A mapping of Quality of Service (QoS) Class Identifier (QCI), resource type, bandwidth options, modulation type, and priority for each slice.
language-python
 Copy code
DEVICES = {
    "eMBB":  ["Smartphone","Tablet","AR Headset","Laptop","Smart TV"],
    "URLLC": ["Autonomous Vehicle","Industrial Robot","Drone","V2X OBU","Remote Surgery Unit"],
    "mMTC":  ["IoT Sensor","Smart Meter","NB-IoT Module","Asset Tracker","Environmental Probe"],
}
Device Types: Lists of devices associated with each slice type.
language-python
 Copy code
USE_CASES = {
    "eMBB":  ["Live Streaming","VR Gaming","Video Conferencing","Cloud Download","AR Navigation"],
    "URLLC": ["V2X Control","Remote Surgery","Industrial Automation","Drone Navigation","Smart Grid"],
    "mMTC":  ["Smart Metering","Asset Tracking","Environmental Monitoring","Fleet Management","Smart Parking"],
}
Use Cases: Potential applications for each slice type.
language-python
 Copy code
rows = []
row_id = 1
Data Storage: Initializes an empty list to store the generated rows and a counter for row IDs.
language-python
 Copy code
for slc in SLICES:
    b   = BASE[slc]
    cfg = SLICE_TYPE_MAP[slc]
Loop Through Slices: Iterates over each slice type, retrieving its base values and configuration.
language-python
 Copy code
    t = np.linspace(0, 10 * np.pi, N)
Time Axis: Creates a time array simulating 10 hours with one sample per minute.
language-python
 Copy code
    load = 50 + 35 * np.sin(t / 3) + 10 * np.sin(t / 1.2)
    load = np.clip(load + np.random.normal(0, 3, N), 10, 99)
Network Load: Generates a sinusoidal load pattern with added noise, simulating network traffic.
language-python
 Copy code
    tp = np.zeros(N)
    tp[0] = b["tp"]
    for i in range(1, N):
        tp[i] = 0.85 * tp[i-1] + 0.15 * b["tp"] * (1 + 0.4 * np.sin(t[i]/4)) \
                + np.random.normal(0, b["tp"] * 0.05)
    tp = np.clip(tp, b["tp"] * 0.3, b["tp"] * 2.5)
Throughput Generation: Initializes an array for throughput and fills it using an AR(1) process based on the load.
language-python
 Copy code
    delay = np.zeros(N)
    delay[0] = b["delay"]
    for i in range(1, N):
        delay[i] = 0.80 * delay[i-1] + 0.20 * b["delay"] * (1 + 0.3 * (load[i]/100)) \
                   + np.random.normal(0, b["delay"] * 0.04)
    delay = np.clip(delay, b["delay"] * 0.5, b["delay"] * 1.8)
Delay Calculation: Similar to throughput, this section generates delay values inversely correlated with load.
language-python
 Copy code
    log_plr = np.zeros(N)
    log_plr[0] = np.log10(b["plr"])
    for i in range(1, N):
        spike = 0.5 if load[i] > 85 else 0.0
        log_plr[i] = 0.88 * log_plr[i-1] + 0.12 * np.log10(b["plr"]) \
                     + spike + np.random.normal(0, 0.08)
    plr = np.clip(10 ** log_plr, 1e-8, 0.1)
Packet Loss Rate (PLR): Generates a log-normal AR(1) process for PLR, introducing spikes during high load.
language-python
 Copy code
    snr    = 30 - 10 * (load / 100) + np.random.normal(0, 1.5, N)
    snr    = np.clip(snr, 5, 35)
    jitter = delay * np.random.uniform(0.03, 0.10, N)
    sig    = -60 - 20 * (load / 100) + np.random.normal(0, 2, N)
    sig    = np.clip(sig, -95, -55)
    num_ues = (load / 100 * (500 if slc == "mMTC" else 80)).astype(int) + \
              np.random.randint(1, 5, N)
    bw_arr  = np.random.choice(cfg["bw"], N)
    gbr_arr = tp * np.where(cfg["rtype"] != "Non-GBR",
                             np.random.uniform(0.5, 0.9, N), 0.0)
Secondary Features: Calculates additional features such as Signal-to-Noise Ratio (SNR), jitter, signal strength, number of users, bandwidth, and guaranteed bit rate.
language-python
 Copy code
    for i in range(N):
        rows.append({
            "Row_ID":                   row_id,
            "Slice_Type":               slc,
            "5QI_Class":                int(np.random.choice(cfg["qci"])),
            "Resource_Type":            cfg["rtype"],
            "QoS_Priority":             cfg["pri"] + np.random.randint(-5, 5),
            "Use_Case":                 np.random.choice(USE_CASES[slc]),
            "Device_Type":              np.random.choice(DEVICES[slc]),
            "Packet_Delay_Budget_ms":   round(float(delay[i]), 4),
            "Packet_Loss_Rate":         float(plr[i]),
            "Throughput_Mbps":          round(float(tp[i]), 6),
            "Guaranteed_Bit_Rate_Mbps": round(float(gbr_arr[i]), 6),
            "Bandwidth_MHz":            int(bw_arr[i]),
            "Signal_Strength_dBm":      round(float(sig[i]), 2),
            "SNR_dB":                   round(float(snr[i]), 2),
            "Jitter_ms":                round(float(jitter[i]), 4),
            "Network_Load_Percent":     round(float(load[i]), 2),
            "Duration_s":               round(float(np.random.uniform(0.5, 120)), 2),
            "Modulation_Type":          cfg["mod"],
            "Num_UEs":                  int(num_ues[i]),
        })
        row_id += 1
Data Collection: Appends each generated row of data to the rows list, including various attributes related to the network slice.
language-python
 Copy code
df = pd.DataFrame(rows)
df.to_csv("dataset.csv", index=False)
DataFrame Creation: Converts the list of rows into a Pandas DataFrame and exports it to a CSV file named "dataset.csv".
language-python
 Copy code
print(f"Dataset: {df.shape}")
print(df.groupby("Slice_Type").size())
print(f"Distinct PDB : {df['Packet_Delay_Budget_ms'].nunique()}")
print(f"Distinct PLR : {df['Packet_Loss_Rate'].nunique()}")
print(f"Distinct TP  : {df['Throughput_Mbps'].nunique()}")
Summary Statistics: Prints the shape of the dataset, the count of entries per slice type, and the number of unique values for packet delay budget, packet loss rate, and throughput.
Conclusion
The provided Python code effectively generates a synthetic dataset that simulates network traffic patterns for different slice types. By incorporating temporal correlations and realistic load cycles, this dataset is well-suited for training LSTM models. The structured approach to data generation ensures that the resulting dataset is comprehensive and reflective of real-world scenarios, making it a valuable resource for researchers and practitioners in the field of network performance analysis.

"""