#!/usr/bin/env python3
"""
realtime_ids_model.py

- Uses Scapy to sniff 802.11 frames on a monitor-mode interface
- Aggregates counts per BSSID in TIME_WINDOW (default 10s)
- Loads the trained model (awid_6f_multiclass.joblib) and predicts each window
"""

import time
import joblib
import numpy as np
from scapy.all import sniff, Dot11
from collections import defaultdict

MODEL_PATH = "awid_6f_multiclass.joblib"
FEATURE_META = "feature_columns.json"
INTERFACE = "wlp0s20f3mon"   # set this to your monitor-mode interface (wlp0s20f3mon, etc.)
TIME_WINDOW = 10.0

# Load model and features metadata
clf = joblib.load(MODEL_PATH)
import json
with open(FEATURE_META, "r") as f:
    meta = json.load(f)
FEATURE_ORDER = meta["features"]

print("Loaded model:", MODEL_PATH)
print("Feature order:", FEATURE_ORDER)
print("Using interface:", INTERFACE)
print("Window (s):", TIME_WINDOW)

# Buffers: per-bssid event lists
buffers = defaultdict(lambda: defaultdict(list))
window_start = time.time()

def handle_packet(pkt):
    global window_start
    if not pkt.haslayer(Dot11):
        return
    dot11 = pkt[Dot11]
    # we need subtype to detect management frames like deauth (12), disassoc(10), probe(4), beacon(8), assoc 0/1/2/3
    try:
        st = int(dot11.subtype)
    except Exception:
        return
    bssid = dot11.addr3 or "00:00:00:00:00:00"
    sa = dot11.addr2 or "00:00:00:00:00:00"

    # classify into our buckets
    if st == 12:
        buffers[bssid]["deauth"].append(sa)
    elif st == 10:
        buffers[bssid]["disassoc"].append(sa)
    elif st == 4:
        buffers[bssid]["probe"].append(sa)
    elif st == 8:
        buffers[bssid]["beacon"].append(sa)
    elif st in (0,1,2,3):
        buffers[bssid]["assoc"].append(sa)

    # check window expiry
    now = time.time()
    if now - window_start >= TIME_WINDOW:
        analyze_and_reset()
        window_start = now

def analyze_and_reset():
    # for each bssid, compute features and predict
    for bssid, data in list(buffers.items()):
        deauth = len(data.get("deauth", []))
        disassoc = len(data.get("disassoc", []))
        probe = len(data.get("probe", []))
        beacon = len(data.get("beacon", []))
        assoc = len(data.get("assoc", []))
        uniq_assoc = len(set(data.get("assoc", [])))

        feat_vec = [deauth, disassoc, probe, beacon, assoc, uniq_assoc]
        # ensure correct order (our training used exactly these columns)
        X = np.array([feat_vec])

        try:
            pred = clf.predict(X)[0]
        except Exception as e:
            print("Prediction error:", e)
            pred = "error"

        ts = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        print(f"[{ts}] BSSID {bssid} → {pred.upper()} (d={deauth},di={disassoc},p={probe},b={beacon},a={assoc},u={uniq_assoc})")
        # optional: write to log file
        with open("realtime_alerts.log", "a") as lf:
            lf.write(json.dumps({
                "timestamp": ts, "bssid": bssid, "prediction": str(pred),
                "features": {"deauth":deauth,"disassoc":disassoc,"probe":probe,"beacon":beacon,"assoc":assoc,"uniq_assoc":uniq_assoc}
            }) + "\n")
    # clear buffers
    buffers.clear()

def main():
    print("Starting realtime IDS sniffing on", INTERFACE)
    print("CTRL-C to stop")
    sniff(iface=INTERFACE, prn=handle_packet, store=False)

if __name__ == "__main__":
    import json
    main()

