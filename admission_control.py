# ============================================================
#  admission_control.py
#  Latency-Based Admission Control
#
#  A slice is ADMITTED if:
#    1. Predicted delay  ≤ ADMISSION_DELAY_THRESH[s]
#    2. Predicted PLR    ≤ LOSS_MAX[s]  (with 10% tolerance buffer)
#
#  The 10% buffer on LOSS_MAX prevents floating-point precision
#  rejections (e.g. PLR=1.03e-5 vs threshold=1.00e-5).
# ============================================================

from config import SLICES, ADMISSION_DELAY_THRESH, LOSS_MAX


def run_admission_control(demands: dict) -> dict:
    """
    Parameters
    ----------
    demands : {slice: {Throughput_Mbps, Packet_Delay_Budget_ms,
                        Packet_Loss_Rate}}
    Returns
    -------
    ac_result : {slice: {admitted, reason, predicted_delay,
                          predicted_loss, predicted_tp}}
    """
    ac_result = {}
    print("\n── Admission Control (Latency-Based) ──")

    for slc in SLICES:
        d     = demands[slc]
        delay = d["Packet_Delay_Budget_ms"]
        loss  = d["Packet_Loss_Rate"]
        tp    = d["Throughput_Mbps"]

        delay_ok = delay <= ADMISSION_DELAY_THRESH[slc]
        loss_ok  = loss  <= LOSS_MAX[slc]           # LOSS_MAX already has 10% buffer

        admitted = delay_ok and loss_ok

        if not delay_ok:
            reason = (f"REJECTED – Predicted delay {delay:.2f} ms exceeds "
                      f"threshold {ADMISSION_DELAY_THRESH[slc]} ms")
        elif not loss_ok:
            reason = (f"REJECTED – Predicted PLR {loss:.2e} exceeds "
                      f"max {LOSS_MAX[slc]:.2e}")
        else:
            reason = "ADMITTED"

        ac_result[slc] = {
            "admitted":        admitted,
            "reason":          reason,
            "predicted_delay": delay,
            "predicted_loss":  loss,
            "predicted_tp":    tp,
        }

        status = "✅ ADMITTED" if admitted else "❌ REJECTED"
        print(f"  [{slc}]  {status}  | delay={delay:.2f} ms  "
              f"PLR={loss:.2e}  TP={tp:.4f} Mbps")
        if not admitted:
            print(f"          → {reason}")

    return ac_result