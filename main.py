#!/usr/bin/env python3
# ============================================================
#  main.py  –  Full Pipeline Orchestrator
#
#  Project: Deep Learning Based Parameter Prediction and
#           Resource Management for Network Slicing (5G)
#
#  Pipeline:
#   1. Generate / load temporal dataset
#   2. Preprocess (log-PLR, sequences)
#   3. Train LSTM per slice → predict Ds(t+1)
#   4. Admission Control (latency + PLR based)
#   5. Resource Allocation: Greedy | Convex | DRL (DQN)
#   6. Plots + Excel results report
#
#  Run:  python main.py
# ============================================================

import os, sys, time, warnings
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

print("=" * 62)
print("  5G NETWORK SLICING — FULL PIPELINE")
print("=" * 62)

# ── STEP 0: Generate temporal dataset (if not already present) ──
DATASET = "dataset.csv"
if not os.path.exists(DATASET):
    print("\n[STEP 0] Generating temporal dataset …")
    import generate_temporal_dataset   # runs on import
else:
    print(f"\n[STEP 0] Dataset found: {DATASET}")

# ── STEP 1: Preprocess ──────────────────────────────────────────
print("\n[STEP 1] Loading & Preprocessing …")
from preprocessing import prepare_all_slices
slice_data, df = prepare_all_slices(DATASET)

# ── STEP 2: Train LSTMs ─────────────────────────────────────────
print("\n[STEP 2] Training LSTM Models (one per slice) …")
from lstm_model import train_all_lstms
t0 = time.time()
models, histories, metrics_all, demands, preds_plot = train_all_lstms(slice_data)
print(f"\n  Total LSTM training time: {time.time()-t0:.1f}s")

print("\n  Predicted Demands Ds(t+1):")
for slc, d in demands.items():
    print(f"    {slc:6s}  TP={d['Throughput_Mbps']:.4f} Mbps  "
          f"Delay={d['Packet_Delay_Budget_ms']:.2f} ms  "
          f"PLR={d['Packet_Loss_Rate']:.2e}")
    


    # ── STEP 2.5: GenAI Recommendation ─────────────────────
print("\n[STEP 2.5] GenAI Recommendation Engine ...")

from genai_recommendation import generate_recommendation
from slice_selector import select_slice

genai_output = generate_recommendation(demands)

for slc, result in genai_output.items():

    print(f"\n  {slc}")

    print(f"     Risk Level     : {result['risk']}")

    print(f"     Recommendation : {result['recommendation']}")

    tp = demands[slc]["Throughput_Mbps"]
    delay = demands[slc]["Packet_Delay_Budget_ms"]
    plr = demands[slc]["Packet_Loss_Rate"]

    selected_slice = select_slice(tp, delay, plr)

    print(f"     AI Selected Slice : {selected_slice}")

# ── STEP 3: Admission Control ────────────────────────────────────
print("\n[STEP 3] Admission Control …")
from admission_control import run_admission_control
ac_result = run_admission_control(demands)

# ── STEP 4: Resource Allocation ──────────────────────────────────
print("\n[STEP 4] Resource Allocation (Greedy | Convex | DRL) …")
from resource_allocation import run_all_methods
alloc_results, snr_db, drl_rewards = run_all_methods(demands, df)

# ── STEP 5: Plots + Excel report ─────────────────────────────────
print("\n[STEP 5] Generating Plots & Excel Report …")
from results import generate_all_plots, save_excel_report

generate_all_plots(
    histories    = histories,
    preds_plot   = preds_plot,
    metrics_all  = metrics_all,
    alloc_results= alloc_results,
    drl_rewards  = drl_rewards,
    ac_result    = ac_result,
    demands      = demands,
    snr_db       = snr_db,
    out_dir      = ".",
)
save_excel_report(
    demands      = demands,
    ac_result    = ac_result,
    alloc_results= alloc_results,
    metrics_all  = metrics_all,
    snr_db       = snr_db,
    path         = "Results_Summary.xlsx",
)

# ── Final Summary ────────────────────────────────────────────────
print("\n" + "=" * 62)
print("  FINAL RESULTS SUMMARY")
print("=" * 62)

from preprocessing import TARGET_COLS_REAL
print("\n── LSTM R² Scores ──")
for slc in ["eMBB", "URLLC", "mMTC"]:
    r2s = [f"{metrics_all[slc][c]['R2']:.4f}" for c in TARGET_COLS_REAL]
    print(f"  {slc:6s}  TP={r2s[0]}  Delay={r2s[1]}  PLR={r2s[2]}")

print("\n── Resource Allocation Results ──")
for meth, res in alloc_results.items():
    viols = "No violations" if not res["violations"] else \
            f"{len(res['violations'])} violation(s)"
    print(f"  {meth:8s}  ∑Us={res['total_utility']:8.2f}  "
          f"RBs={res['rbs']}  QoS: {viols}")

print("\n── Admission Control ──")
for slc in ["eMBB", "URLLC", "mMTC"]:
    status = "✅ ADMITTED" if ac_result[slc]["admitted"] else "❌ REJECTED"
    print(f"  {slc:6s}  {status}")

print("\n✅  Pipeline complete. All outputs saved.")
print("=" * 62)
