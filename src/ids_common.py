"""Shared definitions for the NSL-KDD based Intrusion Detection System."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# 41 traffic features of the NSL-KDD / KDD Cup 99 dataset.
FEATURES = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]
CATEGORICAL = ["protocol_type", "service", "flag"]
NUMERIC = [f for f in FEATURES if f not in CATEGORICAL]
ALL_COLUMNS = FEATURES + ["label", "difficulty"]

# Map each specific attack name to one of the four classic attack families.
ATTACK_FAMILY = {
    "normal": "Normal",
    # Denial of Service
    "back": "DoS", "land": "DoS", "neptune": "DoS", "pod": "DoS", "smurf": "DoS",
    "teardrop": "DoS", "apache2": "DoS", "udpstorm": "DoS", "processtable": "DoS",
    "mailbomb": "DoS",
    # Probing / scanning
    "satan": "Probe", "ipsweep": "Probe", "nmap": "Probe", "portsweep": "Probe",
    "mscan": "Probe", "saint": "Probe",
    # Remote to Local
    "guess_passwd": "R2L", "ftp_write": "R2L", "imap": "R2L", "phf": "R2L",
    "multihop": "R2L", "warezmaster": "R2L", "warezclient": "R2L", "spy": "R2L",
    "xlock": "R2L", "xsnoop": "R2L", "snmpguess": "R2L", "snmpgetattack": "R2L",
    "httptunnel": "R2L", "sendmail": "R2L", "named": "R2L", "worm": "R2L",
    # User to Root
    "buffer_overflow": "U2R", "loadmodule": "U2R", "rootkit": "U2R", "perl": "U2R",
    "sqlattack": "U2R", "xterm": "U2R", "ps": "U2R",
}

FAMILY_DESCRIPTION = {
    "Normal": "Legitimate traffic",
    "DoS": "Denial of Service - flooding a host to exhaust its resources",
    "Probe": "Probing - scanning the network for hosts, ports and weaknesses",
    "R2L": "Remote to Local - unauthorised access from a remote machine",
    "U2R": "User to Root - privilege escalation by a local user",
}

MODEL_FILENAME = "ids_model.joblib"


def resource_dir() -> Path:
    """Folder holding bundled resources, both in source and PyInstaller builds."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def load_records(path) -> pd.DataFrame:
    """Load an NSL-KDD style file.

    Accepts files with or without a header row, and with 41 (features only),
    42 (features + label) or 43 (features + label + difficulty) columns.
    """
    df = pd.read_csv(path, header=None, low_memory=False)
    first = [str(v).strip().lower() for v in df.iloc[0].tolist()]
    if "protocol_type" in first or "duration" in first:
        df.columns = first
        df = df.iloc[1:].reset_index(drop=True)
    else:
        n = df.shape[1]
        if n not in (41, 42, 43):
            raise ValueError(
                f"Expected 41-43 columns (NSL-KDD format), found {n}.")
        df.columns = ALL_COLUMNS[:n]

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing[:8])}")

    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df[NUMERIC] = df[NUMERIC].fillna(0)
    for col in CATEGORICAL:
        df[col] = df[col].astype(str).str.strip()
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")
    return df


def to_family(labels: pd.Series) -> pd.Series:
    return labels.map(lambda l: ATTACK_FAMILY.get(str(l).lower(), "Unknown"))


def parse_single_record(text: str) -> pd.DataFrame:
    """Parse one comma separated connection record (41+ values)."""
    values = [v.strip() for v in text.strip().split(",")]
    if len(values) < 41:
        raise ValueError(f"A record needs 41 comma-separated values, got {len(values)}.")
    row = dict(zip(FEATURES, values[:41]))
    df = pd.DataFrame([row])
    for col in NUMERIC:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[NUMERIC].isna().any(axis=None):
        bad = [c for c in NUMERIC if df[c].isna().any()]
        raise ValueError(f"Non-numeric value in: {', '.join(bad[:5])}")
    return df.astype({c: np.float64 for c in NUMERIC})
