#!/usr/bin/env python3
"""
train_awid_light.py

- Reads headerless AWID CSV (awid.csv) with 155 columns (no header)
- Assigns AWID column names (in correct order)
- Extracts a small set of key fields, aggregates into 10s windows per BSSID
- Produces features.csv (optional) and trains RandomForest classifier
- Saves model to awid_6f_multiclass.joblib and feature_columns.json
"""

import pandas as pd
import numpy as np
import joblib
import json
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# --------------- CONFIG ---------------
DATA_PATH = "awid.csv"   # headerless CSV (155 fields then class)
MODEL_OUT = "awid_6f_multiclass.joblib"
FEATURES_OUT = "features_6f.csv"   # optional (comment out if not desired)
FEATURE_COLUMNS_JSON = "feature_columns.json"
CHUNK_ROWS = 200000      # adjust based on RAM; smaller if memory is low
WINDOW_SIZE = 10.0       # seconds

# --------------- AWID FIELD NAMES (155) ---------------
# IMPORTANT: This list must match the exact order of fields in your AWID CSV.
# I include the common AWID/Wireshark fields in order (truncated visually here).
ALL_COLUMNS = [
    "frame.interface_id","frame.dlt","frame.offset_shift","frame.time_epoch",
    "frame.time_delta","frame.time_delta_displayed","frame.time_relative",
    "frame.len","frame.cap_len","frame.marked","frame.ignored",
    "radiotap.version","radiotap.pad","radiotap.length","radiotap.present.tsft",
    "radiotap.present.flags","radiotap.present.rate","radiotap.present.channel",
    "radiotap.present.fhss","radiotap.present.dbm_antsignal","radiotap.present.dbm_antnoise",
    "radiotap.present.lock_quality","radiotap.present.tx_attenuation",
    "radiotap.present.db_tx_attenuation","radiotap.present.dbm_tx_power",
    "radiotap.present.antenna","radiotap.present.db_antsignal","radiotap.present.db_antnoise",
    "radiotap.present.rxflags","radiotap.present.xchannel","radiotap.present.mcs",
    "radiotap.present.ampdu","radiotap.present.vht","radiotap.present.reserved",
    "radiotap.present.rtap_ns","radiotap.present.vendor_ns","radiotap.present.ext",
    "radiotap.mactime","radiotap.flags.cfp","radiotap.flags.preamble","radiotap.flags.wep",
    "radiotap.flags.frag","radiotap.flags.fcs","radiotap.flags.datapad","radiotap.flags.badfcs",
    "radiotap.flags.shortgi","radiotap.datarate","radiotap.channel.freq","radiotap.channel.type.turbo",
    "radiotap.channel.type.cck","radiotap.channel.type.ofdm","radiotap.channel.type.2ghz",
    "radiotap.channel.type.5ghz","radiotap.channel.type.passive","radiotap.channel.type.dynamic",
    "radiotap.channel.type.gfsk","radiotap.channel.type.gsm","radiotap.channel.type.sturbo",
    "radiotap.channel.type.half","radiotap.channel.type.quarter","radiotap.dbm_antsignal",
    "radiotap.antenna","radiotap.rxflags.badplcp","wlan.fc.type_subtype","wlan.fc.version",
    "wlan.fc.type","wlan.fc.subtype","wlan.fc.ds","wlan.fc.frag","wlan.fc.retry",
    "wlan.fc.pwrmgt","wlan.fc.moredata","wlan.fc.protected","wlan.fc.order",
    "wlan.duration","wlan.ra","wlan.da","wlan.ta","wlan.sa","wlan.bssid","wlan.frag",
    "wlan.seq","wlan.bar.type","wlan.ba.control.ackpolicy","wlan.ba.control.multitid",
    "wlan.ba.control.cbitmap","wlan.bar.compressed.tidinfo","wlan.ba.bm","wlan.fcs_good",
    "wlan_mgt.fixed.capabilities.ess","wlan_mgt.fixed.capabilities.ibss",
    "wlan_mgt.fixed.capabilities.cfpoll.ap","wlan_mgt.fixed.capabilities.privacy",
    "wlan_mgt.fixed.capabilities.preamble","wlan_mgt.fixed.capabilities.pbcc",
    "wlan_mgt.fixed.capabilities.agility","wlan_mgt.fixed.capabilities.spec_man",
    "wlan_mgt.fixed.capabilities.short_slot_time","wlan_mgt.fixed.capabilities.apsd",
    "wlan_mgt.fixed.capabilities.radio_measurement","wlan_mgt.fixed.capabilities.dsss_ofdm",
    "wlan_mgt.fixed.capabilities.del_blk_ack","wlan_mgt.fixed.capabilities.imm_blk_ack",
    "wlan_mgt.fixed.listen_ival","wlan_mgt.fixed.current_ap","wlan_mgt.fixed.status_code",
    "wlan_mgt.fixed.timestamp","wlan_mgt.fixed.beacon","wlan_mgt.fixed.aid",
    "wlan_mgt.fixed.reason_code","wlan_mgt.fixed.auth.alg","wlan_mgt.fixed.auth_seq",
    "wlan_mgt.fixed.category_code","wlan_mgt.fixed.htact","wlan_mgt.fixed.chanwidth",
    "wlan_mgt.fixed.fragment","wlan_mgt.fixed.sequence","wlan_mgt.tagged.all","wlan_mgt.ssid",
    "wlan_mgt.ds.current_channel","wlan_mgt.tim.dtim_count","wlan_mgt.tim.dtim_period",
    "wlan_mgt.tim.bmapctl.multicast","wlan_mgt.tim.bmapctl.offset","wlan_mgt.country_info.environment",
    "wlan_mgt.rsn.version","wlan_mgt.rsn.gcs.type","wlan_mgt.rsn.pcs.count",
    "wlan_mgt.rsn.akms.count","wlan_mgt.rsn.akms.type","wlan_mgt.rsn.capabilities.preauth",
    "wlan_mgt.rsn.capabilities.no_pairwise","wlan_mgt.rsn.capabilities.ptksa_replay_counter",
    "wlan_mgt.rsn.capabilities.gtksa_replay_counter","wlan_mgt.rsn.capabilities.mfpr",
    "wlan_mgt.rsn.capabilities.mfpc","wlan_mgt.rsn.capabilities.peerkey","wlan_mgt.tcprep.trsmt_pow",
    "wlan_mgt.tcprep.link_mrg","wlan.wep.iv","wlan.wep.key","wlan.wep.icv","wlan.tkip.extiv",
    "wlan.ccmp.extiv","wlan.qos.tid","wlan.qos.priority","wlan.qos.eosp","wlan.qos.ack",
    "wlan.qos.amsdupresent","wlan.qos.buf_state_indicated","wlan.qos.bit4","wlan.qos.txop_dur_req",
    "wlan.qos.buf_state_indicated","data.len","class"
]

# Quick helper - which columns we will use (names)
USE_COLS = [
    "frame.time_relative",
    "wlan.fc.type_subtype",
    "wlan.sa",
    "wlan.bssid",
    "radiotap.present.dbm_antsignal" if "radiotap.present.dbm_antsignal" in ALL_COLUMNS else "radiotap.dbm_antsignal",
    "frame.len",
    "class"
]

# Validate that the names exist in ALL_COLUMNS
for c in USE_COLS:
    if c not in ALL_COLUMNS:
        print("WARNING: requested column not found in ALL_COLUMNS:", c)
        # We will still try to proceed; mapping by index will fail if missing.

# Compute indices for usecols
use_indices = [ALL_COLUMNS.index(c) for c in USE_COLS if c in ALL_COLUMNS]
use_names = [c for c in USE_COLS if c in ALL_COLUMNS]

print("Using columns (by name):", use_names)
print("Using columns (by index):", use_indices)

# If no columns found, error out
if len(use_indices) < 4:
    raise SystemExit("Not enough usable columns found in ALL_COLUMNS mapping. Check names list.")

# --- helper: aggregate chunk into windows per bssid ---
def aggregate_chunk(df_chunk):
    # df_chunk: DataFrame with columns named by use_names (we will ensure names on read)
    # cleaning
    df = df_chunk.copy()
    # replace '?' with NaN, then drop rows missing key fields
    df.replace('?', np.nan, inplace=True)
    needed = ["frame.time_relative", "wlan.fc.type_subtype", "wlan.sa", "wlan.bssid", "class"]
    for k in needed:
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors='coerce') if k.startswith("frame.") or k.startswith("radiotap") else df[k]
    df.dropna(subset=["frame.time_relative", "wlan.fc.type_subtype", "wlan.sa", "wlan.bssid", "class"], inplace=True)

    if df.empty:
        return pd.DataFrame(columns=[
            "deauth_count","disassoc_count","probe_req_count","beacon_count","assoc_count","unique_assoc_srcs","label"
        ])

    df = df.sort_values("frame.time_relative")
    tmin = df["frame.time_relative"].min()
    tmax = df["frame.time_relative"].max()
    windows = np.arange(tmin, tmax + 1e-9, WINDOW_SIZE)

    rows = []
    for w0 in windows:
        w1 = w0 + WINDOW_SIZE
        w = df[(df["frame.time_relative"] >= w0) & (df["frame.time_relative"] < w1)]
        if w.empty:
            continue
        # group by bssid
        for bssid, g in w.groupby("wlan.bssid"):
            subtypes = g["wlan.fc.type_subtype"].astype(int)
            deauth = (subtypes == 12).sum()
            disassoc = (subtypes == 10).sum()
            probe = (subtypes == 4).sum()
            beacon = (subtypes == 8).sum()
            assoc = g["wlan.fc.type_subtype"].isin([0,1,2,3]).sum()
            uniq_assoc = g[g["wlan.fc.type_subtype"].isin([0,1,2,3])]["wlan.sa"].nunique()

            # label priority
            labs = g["class"].astype(str).str.lower().unique()
            if any("flood" in s for s in labs):
                label = "flooding"
            elif any("imperson" in s for s in labs):
                label = "impersonation"
            elif any("inject" in s for s in labs):
                label = "injection"
            else:
                label = "normal"

            rows.append([deauth, disassoc, probe, beacon, assoc, uniq_assoc, label])

    df_out = pd.DataFrame(rows, columns=[
        "deauth_count","disassoc_count","probe_req_count","beacon_count","assoc_count","unique_assoc_srcs","label"
    ])
    return df_out

# --- main processing loop: read in chunks and aggregate ---
all_features = []
start = time.time()
reader = pd.read_csv(DATA_PATH, header=None, usecols=use_indices, names=use_names, chunksize=CHUNK_ROWS, low_memory=False)

cnt = 0
for chunk in reader:
    cnt += 1
    print(f"Processing chunk #{cnt} rows ~ {len(chunk)}")
    feats = aggregate_chunk(chunk)
    if not feats.empty:
        all_features.append(feats)
    # optional: limit for quick testing (uncomment)
    # if cnt >= 4: break

if len(all_features) == 0:
    raise SystemExit("No feature windows extracted. Check DATA_PATH and column mapping.")

features_df = pd.concat(all_features, ignore_index=True)
print("Total windows extracted:", len(features_df))
if FEATURES_OUT:
    features_df.to_csv(FEATURES_OUT, index=False)
    print("Saved aggregated features to", FEATURES_OUT)

# --- train classifier ---
X = features_df.drop(columns=["label"])
y = features_df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("Training samples:", len(X_train), "Test samples:", len(X_test))

clf = RandomForestClassifier(n_estimators=200, class_weight="balanced", n_jobs=-1, random_state=42)
print("Training RandomForest...")
clf.fit(X_train, y_train)

print("Evaluating on test set...")
preds = clf.predict(X_test)
print(classification_report(y_test, preds))

# save model + feature column ordering
joblib.dump(clf, MODEL_OUT)
with open(FEATURE_COLUMNS_JSON, "w") as f:
    json.dump({
        "features": X.columns.tolist(),
        "window_size": WINDOW_SIZE
    }, f)
print("Saved model to", MODEL_OUT)
print("Saved feature meta to", FEATURE_COLUMNS_JSON)
print("Total time:", time.time() - start, "s")

