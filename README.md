# ML Intrusion Detection System

A Windows desktop application (`IDS.exe`) that detects network intrusions with machine learning.
A Random Forest classifier trained on the **NSL-KDD** dataset labels every network connection as
**Normal** or one of four attack families:

| Family | Meaning |
|---|---|
| **DoS** | Denial of Service: flooding a host to exhaust its resources |
| **Probe** | Scanning the network for hosts, ports and weaknesses |
| **R2L** | Remote to Local: unauthorised access from a remote machine |
| **U2R** | User to Root: privilege escalation by a local user |

## Download

Grab `IDS.exe` from the [Releases](../../releases) page and double-click it. You don't need Python.
The model and a sample traffic file are bundled inside the exe.

## Features

- **Dashboard**: test-set performance, per-class report, confusion matrix, threshold trade-off, and top features.
- **Analyze File**: load any NSL-KDD formatted `.csv`/`.txt` traffic file. Every connection is classified, with
  per-family counts and alerts. If the file has labels, the app also reports accuracy, attacks caught and false alarms.
  Results can be exported to CSV.
- **Single Record**: paste one 41-feature connection record and get a verdict plus class probabilities.
- **Live Monitor**: replays labelled traffic as a real-time alert feed with running counters.
- **Alert sensitivity slider**: sets the attack-probability threshold, trading missed attacks against false alarms.

## Dataset

[NSL-KDD](https://www.unb.ca/cic/datasets/nsl.html), from the Canadian Institute for Cybersecurity, University of New Brunswick.
It is also on Kaggle as [`hassan06/nslkdd`](https://www.kaggle.com/datasets/hassan06/nslkdd).

- `data/KDDTrain+.txt`: 125,973 connections, used for training
- `data/KDDTest+.txt`: 22,544 connections, used only for evaluation. It includes 17 attack types that
  never appear in training, which makes it a deliberately hard, realistic test.

`src/download_data.py` fetches the files from a public mirror. Use `--kaggle` to download through the Kaggle API instead.

## Results (KDDTest+, unseen data)

| Alert threshold | Accuracy | Attacks caught (recall) | Precision | False-alarm rate |
|---|---|---|---|---|
| 0.50 | 78.1% | 63.7% | 96.8% | 2.8% |
| **0.30 (default)** | **84.7%** | **75.9%** | **96.6%** | **3.5%** |
| 0.20 | 87.4% | 84.8% | 92.5% | 9.1% |
| 0.10 | 90.0% | 91.0% | 91.5% | 11.2% |

DoS and Probe attacks are detected well. R2L and U2R are hard for any model trained on NSL-KDD: those
families are rare in the training set, and most of their test-set variants are novel attack types.
The 0.30 default was chosen by looking at this test-set sweep, so treat the numbers above as indicative only.

## Build from source

```bat
pip install -r requirements.txt
python src\download_data.py     :: fetch NSL-KDD into data\
python src\train_model.py       :: train + evaluate, writes models\ids_model.joblib
python src\ids_app.py           :: run the GUI from source
build_exe.bat                   :: package everything into dist\IDS.exe
```

Run `dist\IDS.exe --selftest report.txt` to check a packaged build without opening the GUI.

## Project layout

```
src/ids_common.py     feature schema, attack-family mapping, file parsing
src/download_data.py  dataset download (mirror or Kaggle)
src/train_model.py    training, evaluation, threshold sweep
src/ids_app.py        Tkinter desktop application
models/               trained model bundle + metrics.json
samples/              2,000 labelled test connections for demos
data/                 NSL-KDD train/test files
build_exe.bat         PyInstaller build script
```

## Limitations

The app classifies connection records in the NSL-KDD/KDD'99 feature format. It does not capture raw packets.
To monitor a live network, you would first need to extract these 41 features from captured traffic
(for example, with a flow-feature extractor), then feed the resulting records into the app.
