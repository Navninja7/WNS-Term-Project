# WNS-Term-Project
Light weight Intrusion Detection System 


# Wi-Fi Intrusion Detection System (IDS) using Machine Learning

This project implements a **Machine Learning-based Intrusion Detection System (IDS)** for **Wi-Fi networks** using the **AWID dataset**.  
It detects four major classes of wireless attacks — **Flooding**, **Impersonation**, **Injection**, and **Normal** — by analyzing Wi-Fi management frames captured in real time.

---

## Overview

Traditional signature-based IDS solutions fail to detect evolving Wi-Fi attacks.  
This project introduces a **data-driven ML approach** that classifies Wi-Fi traffic based on key wireless frame features.

We trained a **Random Forest classifier** using selected parameters from the AWID dataset and deployed a **real-time packet sniffer** using **Scapy** to detect malicious frames live from a monitor-mode interface.

---

## Features

- **Offline Training:** Train model on AWID dataset (`awid.csv`) with automatic preprocessing.  
- **Real-time Detection:** Monitor live traffic (via `wlan0mon`) to detect ongoing attacks.  
- **Multi-class Classification:** Detects  
  - Flooding  
  - Impersonation  
  - Injection  
  - Normal  
- **Model Persistence:** Trained models stored via `joblib` for fast reuse.  
- **Lightweight:** Uses only 6 key frame-level features — optimal for resource-limited setups (e.g., Raspberry Pi).

---

## Attack Classes & Features

| Feature | Description | Typical Attack Indicator |
|----------|--------------|---------------------------|
| `deauth_count` | Deauthentication frames per time window | 🔺 Flooding |
| `disassoc_count` | Disassociation frames per time window | 🔺 Flooding |
| `probe_req_count` | Probe requests per window | 🔺 Injection / Flooding |
| `beacon_count` | Beacon frames per window | 🔺 Flooding / Rogue AP |
| `assoc_count` | Association frames per window | 🔺 Impersonation |
| `unique_assoc_srcs` | Unique MAC sources per window | 🔺 Impersonation |

---

## Hardware & Software Setup

**Hardware:**
- Laptop Wi-Fi card supporting **Monitor Mode**
- (Optional) Raspberry Pi 4 as Access Point (for attack testing)

**Software:**
- Ubuntu 22.04 / Kali Linux
- Python 3.8+
- Scapy, Pandas, Scikit-learn, Joblib
- Aircrack-ng suite (for generating attacks)
- AWID Dataset (download from [AWID Repository](https://icsdweb.aegean.gr/awid/))

---

## Installation

```bash
# Clone the repository
git clone https://github.com/<yourusername>/WiFi-IDS.git
cd WiFi-IDS

# Create venv
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
