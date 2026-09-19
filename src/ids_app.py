"""ML Intrusion Detection System - desktop application (Tkinter).

Tabs:
  * Dashboard       - model details, test-set performance, top features
  * Analyze File    - classify every connection in a CSV/TXT traffic file
  * Single Record   - classify one connection record
  * Live Monitor    - replay labelled traffic as a real-time alert feed
"""

import queue
import random
import threading
import tkinter as tk
from collections import Counter
from tkinter import filedialog, messagebox, ttk

import joblib
import numpy as np
import pandas as pd
# The pickled model references these; importing them explicitly lets
# PyInstaller discover and bundle scikit-learn into the .exe.
import sklearn.compose  # noqa: F401
import sklearn.ensemble  # noqa: F401
import sklearn.pipeline  # noqa: F401
import sklearn.preprocessing  # noqa: F401

from ids_common import (FAMILY_DESCRIPTION, FEATURES, MODEL_FILENAME,
                        load_records, parse_single_record, resource_dir,
                        to_family)

APP_TITLE = "ML Intrusion Detection System"
FAMILIES = ["Normal", "DoS", "Probe", "R2L", "U2R"]
COLORS = {"Normal": "#2e9d5b", "DoS": "#d64545", "Probe": "#e0892b",
          "R2L": "#8a4fd1", "U2R": "#c2338f", "Unknown": "#777777"}
MAX_TABLE_ROWS = 5000


class Detector:
    """Wraps the trained model with a tunable attack threshold."""

    def __init__(self, bundle: dict):
        self.bundle = bundle
        self.model = bundle["model"]
        self.classes = np.array(bundle["classes"])
        self.normal_idx = int(np.where(self.classes == "Normal")[0][0])
        self.attack_idx = np.array([i for i in range(len(self.classes)) if i != self.normal_idx])

    def detect(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        proba = self.model.predict_proba(df[FEATURES])
        p_attack = 1.0 - proba[:, self.normal_idx]
        family = self.classes[self.attack_idx][proba[:, self.attack_idx].argmax(axis=1)]
        is_attack = p_attack >= threshold
        out = pd.DataFrame({
            "prediction": np.where(is_attack, family, "Normal"),
            "attack_probability": p_attack.round(4),
            "alert": np.where(is_attack, "ALERT", ""),
        })
        return out


class IDSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1100x720")
        self.minsize(900, 600)
        try:
            self.iconbitmap(str(resource_dir() / "assets" / "ids.ico"))
        except tk.TclError:
            pass

        self.ok = False
        model_path = resource_dir() / "models" / MODEL_FILENAME
        try:
            self.detector = Detector(joblib.load(model_path))
        except Exception as exc:  # noqa: BLE001 - show any load failure to user
            messagebox.showerror(APP_TITLE, f"Could not load model:\n{model_path}\n\n{exc}")
            self.destroy()
            return
        self.ok = True

        self.threshold = tk.DoubleVar(value=self.detector.bundle.get("default_threshold", 0.3))
        self.last_results = None
        self.live_job = None
        self.live_rows = None
        self.live_stats = Counter()

        self._style()
        self._build_header()
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._build_dashboard(nb)
        self._build_file_tab(nb)
        self._build_single_tab(nb)
        self._build_live_tab(nb)
        self.status = tk.StringVar(value="Model loaded. Ready.")
        ttk.Label(self, textvariable=self.status, style="Status.TLabel",
                  anchor="w").pack(fill="x", side="bottom")

    # ------------------------------------------------------------------ UI
    def _style(self):
        s = ttk.Style(self)
        if "vista" in s.theme_names():
            s.theme_use("vista")
        s.configure("Title.TLabel", font=("Segoe UI", 16, "bold"))
        s.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#555")
        s.configure("H.TLabel", font=("Segoe UI", 11, "bold"))
        s.configure("Big.TLabel", font=("Segoe UI", 20, "bold"))
        s.configure("Status.TLabel", font=("Segoe UI", 9), padding=(10, 3))
        s.configure("Treeview", rowheight=22)

    def _build_header(self):
        top = ttk.Frame(self, padding=(12, 10))
        top.pack(fill="x")
        left = ttk.Frame(top)
        left.pack(side="left")
        ttk.Label(left, text="\U0001F6E1  " + APP_TITLE, style="Title.TLabel").pack(anchor="w")
        ttk.Label(left, text="Random Forest classifier trained on the NSL-KDD dataset",
                  style="Sub.TLabel").pack(anchor="w")

        right = ttk.Frame(top)
        right.pack(side="right")
        ttk.Label(right, text="Alert sensitivity (attack probability threshold)").pack(anchor="e")
        row = ttk.Frame(right)
        row.pack(anchor="e")
        self.th_label = ttk.Label(row, width=5, text=f"{self.threshold.get():.2f}")
        ttk.Label(row, text="more alerts").pack(side="left")
        ttk.Scale(row, from_=0.05, to=0.95, variable=self.threshold, length=200,
                  command=lambda v: self.th_label.config(text=f"{float(v):.2f}")
                  ).pack(side="left", padx=4)
        ttk.Label(row, text="fewer alerts").pack(side="left")
        self.th_label.pack(side="left", padx=(8, 0))

    def _build_dashboard(self, nb):
        tab = ttk.Frame(nb, padding=14)
        nb.add(tab, text="  Dashboard  ")
        b = self.detector.bundle
        m = b["metrics"]

        cards = ttk.Frame(tab)
        cards.pack(fill="x")
        th = b.get("default_threshold", 0.3)
        at_default = next((r for r in b.get("threshold_sweep", []) if abs(r["threshold"] - th) < 1e-9), None)
        items = [
            ("Detection accuracy", at_default["accuracy"] if at_default else m["binary_accuracy"]),
            ("Attack recall", at_default["recall"] if at_default else m["attack_recall"]),
            ("Alert precision", at_default["precision"] if at_default else m["attack_precision"]),
            ("False alarm rate", at_default["false_alarm_rate"] if at_default else m["false_alarm_rate"]),
        ]
        for i, (name, val) in enumerate(items):
            f = ttk.LabelFrame(cards, text=name, padding=10)
            f.grid(row=0, column=i, sticky="nsew", padx=5)
            cards.columnconfigure(i, weight=1)
            ttk.Label(f, text=f"{val * 100:.1f}%", style="Big.TLabel").pack()
        ttk.Label(tab, style="Sub.TLabel", text=(
            f"Scores measured on {b['test_rows']:,} unseen KDDTest+ connections at threshold {th:.2f} "
            f"(includes attack types never seen in training).   Trained on {b['train_rows']:,} rows, {b['trained_at']}.")
        ).pack(anchor="w", pady=(6, 10))

        body = ttk.Frame(tab)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3)
        body.columnconfigure(1, weight=2)
        body.rowconfigure(0, weight=1)

        left = ttk.LabelFrame(body, text="Per-class report (argmax prediction)", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        txt = tk.Text(left, font=("Consolas", 10), height=14, wrap="none", relief="flat")
        txt.insert("end", b["report"])
        txt.insert("end", "\nConfusion matrix (rows = true, columns = predicted)\n")
        labels = b["confusion_labels"]
        txt.insert("end", " " * 8 + "".join(f"{l:>8}" for l in labels) + "\n")
        for l, row in zip(labels, b["confusion_matrix"]):
            txt.insert("end", f"{l:>8}" + "".join(f"{v:>8}" for v in row) + "\n")
        txt.insert("end", "\nThreshold trade-off (binary attack detection)\n")
        for r in b.get("threshold_sweep", []):
            txt.insert("end", "  th={threshold:.2f}  acc={accuracy:.3f}  recall={recall:.3f}  "
                              "precision={precision:.3f}  false-alarm={false_alarm_rate:.3f}\n".format(**r))
        txt.config(state="disabled")
        txt.pack(fill="both", expand=True)

        right = ttk.LabelFrame(body, text="Most important features", padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        c = tk.Canvas(right, height=260, highlightthickness=0, bg="white")
        c.pack(fill="both", expand=True)
        top = b["top_features"]

        def draw(_e=None):
            c.delete("all")
            w = max(c.winfo_width(), 300)
            mx = max(v for _, v in top) or 1
            for i, (name, v) in enumerate(top):
                y = 10 + i * 26
                c.create_text(8, y + 9, text=name, anchor="w", font=("Segoe UI", 9))
                bw = (w - 200) * v / mx
                c.create_rectangle(160, y + 2, 160 + bw, y + 18, fill="#3b6fd8", width=0)
                c.create_text(165 + bw, y + 10, text=f"{v:.3f}", anchor="w", font=("Segoe UI", 8))
        c.bind("<Configure>", draw)

        info = ttk.LabelFrame(tab, text="Attack families", padding=8)
        info.pack(fill="x", pady=(10, 0))
        for fam in FAMILIES:
            r = ttk.Frame(info)
            r.pack(anchor="w")
            tk.Label(r, text="  ", bg=COLORS[fam]).pack(side="left", padx=(0, 6))
            ttk.Label(r, text=f"{fam}: {FAMILY_DESCRIPTION[fam]}").pack(side="left")

    def _build_file_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Analyze File  ")

        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Button(bar, text="Open traffic file...", command=self.open_file).pack(side="left")
        ttk.Button(bar, text="Use bundled sample", command=self.open_sample).pack(side="left", padx=6)
        ttk.Button(bar, text="Export results...", command=self.export_results).pack(side="left")
        self.only_alerts = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="Show alerts only", variable=self.only_alerts,
                        command=self.refresh_table).pack(side="left", padx=12)
        self.file_label = ttk.Label(bar, text="No file loaded", style="Sub.TLabel")
        self.file_label.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(bar, mode="indeterminate", length=120)
        self.progress.pack(side="right")

        ttk.Label(tab, style="Sub.TLabel", text=(
            "Expected format: NSL-KDD / KDD'99 connection records - 41 feature columns, optionally followed "
            "by a label column (enables accuracy scoring). Header row optional.")).pack(anchor="w", pady=(6, 4))

        summary = ttk.Frame(tab)
        summary.pack(fill="x", pady=4)
        self.summary_canvas = tk.Canvas(summary, height=120, bg="white", highlightthickness=0)
        self.summary_canvas.pack(side="left", fill="x", expand=True)
        self.summary_text = ttk.Label(summary, text="", justify="left", width=40)
        self.summary_text.pack(side="left", padx=10)

        cols = ("#", "protocol", "service", "flag", "src_bytes", "dst_bytes",
                "prediction", "attack_prob", "true_label")
        frame = ttk.Frame(tab)
        frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(frame, columns=cols, show="headings")
        for col, w in zip(cols, (60, 70, 90, 60, 90, 90, 100, 90, 150)):
            self.tree.heading(col, text=col)
            self.tree.column(col, width=w, anchor="center")
        for fam, color in COLORS.items():
            self.tree.tag_configure(fam, foreground=color)
        sb = ttk.Scrollbar(frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

    def _build_single_tab(self, nb):
        tab = ttk.Frame(nb, padding=14)
        nb.add(tab, text="  Single Record  ")
        ttk.Label(tab, text="Paste one connection record (41 comma-separated NSL-KDD features):",
                  style="H.TLabel").pack(anchor="w")
        self.single_entry = tk.Text(tab, height=4, font=("Consolas", 10), wrap="word")
        self.single_entry.pack(fill="x", pady=6)
        self.single_entry.insert("end", "0,tcp,private,S0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,123,6,1.00,"
                                        "1.00,0.00,0.00,0.05,0.07,0.00,255,26,0.10,0.05,0.00,0.00,1.00,1.00,0.00,0.00")
        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        ttk.Button(bar, text="Analyze", command=self.analyze_single).pack(side="left")
        ttk.Button(bar, text="Random normal sample", command=lambda: self.random_sample(True)).pack(side="left", padx=6)
        ttk.Button(bar, text="Random attack sample", command=lambda: self.random_sample(False)).pack(side="left")
        self.single_truth = ttk.Label(bar, text="", style="Sub.TLabel")
        self.single_truth.pack(side="left", padx=12)

        self.verdict = tk.Label(tab, text="", font=("Segoe UI", 22, "bold"), pady=16)
        self.verdict.pack(fill="x", pady=(18, 4))
        self.verdict_detail = ttk.Label(tab, text="", font=("Segoe UI", 11), justify="left")
        self.verdict_detail.pack(anchor="w")
        self.proba_canvas = tk.Canvas(tab, height=150, bg="white", highlightthickness=0)
        self.proba_canvas.pack(fill="x", pady=10)

    def _build_live_tab(self, nb):
        tab = ttk.Frame(nb, padding=10)
        nb.add(tab, text="  Live Monitor  ")
        bar = ttk.Frame(tab)
        bar.pack(fill="x")
        self.live_btn = ttk.Button(bar, text="▶ Start monitoring", command=self.toggle_live)
        self.live_btn.pack(side="left")
        ttk.Button(bar, text="Clear", command=self.clear_live).pack(side="left", padx=6)
        ttk.Label(bar, text="Speed (connections/sec):").pack(side="left", padx=(16, 4))
        self.live_speed = tk.IntVar(value=10)
        ttk.Spinbox(bar, from_=1, to=100, textvariable=self.live_speed, width=5).pack(side="left")
        ttk.Label(tab, style="Sub.TLabel", text=(
            "Replays labelled NSL-KDD test traffic (or the last analysed file) through the detector as if it "
            "were arriving live. Real packet capture requires feature extraction to the KDD format.")
        ).pack(anchor="w", pady=6)

        stats = ttk.Frame(tab)
        stats.pack(fill="x")
        self.live_labels = {}
        for i, key in enumerate(["Processed", "Alerts"] + FAMILIES[1:] + ["Correct"]):
            f = ttk.LabelFrame(stats, text=key, padding=6)
            f.grid(row=0, column=i, sticky="nsew", padx=3)
            stats.columnconfigure(i, weight=1)
            lbl = tk.Label(f, text="0", font=("Segoe UI", 16, "bold"),
                           fg=COLORS.get(key, "#222"))
            lbl.pack()
            self.live_labels[key] = lbl

        self.live_log = tk.Text(tab, font=("Consolas", 10), bg="#111", fg="#ddd", wrap="none")
        self.live_log.pack(fill="both", expand=True, pady=(8, 0))
        for fam, color in COLORS.items():
            self.live_log.tag_configure(fam, foreground=color if fam != "Normal" else "#6fcf97")

    # -------------------------------------------------------------- File tab
    def open_sample(self):
        self._load_and_analyze(resource_dir() / "samples" / "sample_traffic.csv")

    def open_file(self):
        path = filedialog.askopenfilename(
            title="Select network traffic file",
            filetypes=[("Traffic data", "*.csv *.txt"), ("All files", "*.*")])
        if path:
            self._load_and_analyze(path)

    def _load_and_analyze(self, path):
        self.status.set(f"Analysing {path} ...")
        self.progress.start(12)
        th = self.threshold.get()

        box = queue.Queue()

        def work():  # runs off the UI thread; never touches Tk directly
            try:
                df = load_records(path)
                res = self.detector.detect(df, th)
                out = pd.concat([df.reset_index(drop=True), res], axis=1)
                if "label" in out.columns:
                    out["true_family"] = to_family(out["label"])
                box.put((self._show_results, (out, str(path), th)))
            except Exception as exc:  # noqa: BLE001
                box.put((self._analysis_failed, (exc,)))

        def poll():
            try:
                fn, args = box.get_nowait()
            except queue.Empty:
                self.after(100, poll)
                return
            fn(*args)

        threading.Thread(target=work, daemon=True).start()
        poll()

    def _analysis_failed(self, exc):
        self.progress.stop()
        self.status.set("Analysis failed.")
        messagebox.showerror(APP_TITLE, f"Could not analyse file:\n\n{exc}")

    def _show_results(self, out, path, th):
        self.progress.stop()
        self.last_results = out
        self.file_label.config(text=f"{path}  ({len(out):,} connections, threshold {th:.2f})")
        counts = out["prediction"].value_counts()
        n_alert = int((out["alert"] == "ALERT").sum())
        lines = [f"Connections analysed: {len(out):,}",
                 f"Alerts raised: {n_alert:,} ({n_alert / max(len(out), 1):.1%})"]
        if "true_family" in out.columns:
            truth = out["true_family"] != "Normal"
            flag = out["alert"] == "ALERT"
            lines.append(f"Detection accuracy vs labels: {(truth == flag).mean():.1%}")
            if truth.any():
                lines.append(f"Attacks caught: {(truth & flag).sum():,} / {truth.sum():,}")
            lines.append(f"False alarms: {(~truth & flag).sum():,}")
        self.summary_text.config(text="\n".join(lines))
        self._draw_bars(self.summary_canvas, {f: int(counts.get(f, 0)) for f in FAMILIES})
        self.refresh_table()
        self.status.set(f"Done. {n_alert:,} alerts out of {len(out):,} connections.")

    def refresh_table(self):
        self.tree.delete(*self.tree.get_children())
        out = self.last_results
        if out is None:
            return
        view = out[out["alert"] == "ALERT"] if self.only_alerts.get() else out
        for idx, r in view.head(MAX_TABLE_ROWS).iterrows():
            self.tree.insert("", "end", tags=(r["prediction"],), values=(
                idx + 1, r["protocol_type"], r["service"], r["flag"], int(r["src_bytes"]),
                int(r["dst_bytes"]), r["prediction"], f"{r['attack_probability']:.3f}",
                r.get("label", "") if "label" in out.columns else ""))
        if len(view) > MAX_TABLE_ROWS:
            self.status.set(f"Showing first {MAX_TABLE_ROWS:,} of {len(view):,} rows - export to see all.")

    def export_results(self):
        if self.last_results is None:
            messagebox.showinfo(APP_TITLE, "Analyse a file first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".csv", initialfile="ids_results.csv",
                                            filetypes=[("CSV", "*.csv")])
        if path:
            self.last_results.to_csv(path, index=False)
            self.status.set(f"Results exported to {path}")

    def _draw_bars(self, canvas, data):
        canvas.update_idletasks()
        canvas.delete("all")
        w = max(canvas.winfo_width(), 400)
        total = max(sum(data.values()), 1)
        mx = max(max(data.values()), 1)
        for i, (fam, v) in enumerate(data.items()):
            y = 6 + i * 22
            canvas.create_text(8, y + 8, text=fam, anchor="w", font=("Segoe UI", 9, "bold"))
            bw = (w - 190) * v / mx
            canvas.create_rectangle(70, y, 70 + bw, y + 16, fill=COLORS[fam], width=0)
            canvas.create_text(76 + bw, y + 8, anchor="w", font=("Segoe UI", 9),
                               text=f"{v:,} ({v / total:.1%})")

    # ------------------------------------------------------------ Single tab
    def _sample_frame(self):
        if self.live_rows is None:
            self.live_rows = load_records(resource_dir() / "samples" / "sample_traffic.csv")
        return self.live_rows

    def random_sample(self, normal: bool):
        df = self._sample_frame()
        fam = to_family(df["label"])
        pool = df[(fam == "Normal") if normal else (fam != "Normal")]
        row = pool.iloc[random.randrange(len(pool))]
        text = ",".join(str(row[f]) if not isinstance(row[f], float) or not row[f].is_integer()
                        else str(int(row[f])) for f in FEATURES)
        self.single_entry.delete("1.0", "end")
        self.single_entry.insert("end", text)
        self.single_truth.config(text=f"True label: {row['label']} ({to_family(pd.Series([row['label']]))[0]})")
        self.analyze_single(keep_truth=True)

    def analyze_single(self, keep_truth=False):
        if not keep_truth:
            self.single_truth.config(text="")
        try:
            df = parse_single_record(self.single_entry.get("1.0", "end"))
        except ValueError as exc:
            messagebox.showerror(APP_TITLE, str(exc))
            return
        th = self.threshold.get()
        res = self.detector.detect(df, th).iloc[0]
        fam = res["prediction"]
        if fam == "Normal":
            self.verdict.config(text="✔  NORMAL TRAFFIC", bg="#e3f5ea", fg=COLORS["Normal"])
        else:
            self.verdict.config(text=f"⚠  INTRUSION DETECTED - {fam}", bg="#fbe4e4", fg=COLORS[fam])
        self.verdict_detail.config(text=(
            f"{FAMILY_DESCRIPTION[fam]}\n"
            f"Attack probability: {res['attack_probability']:.1%}   (alert threshold {th:.2f})"))
        proba = self.detector.model.predict_proba(df[FEATURES])[0]
        self.proba_canvas.delete("all")
        w = max(self.proba_canvas.winfo_width(), 400)
        for i, (c, p) in enumerate(zip(self.detector.classes, proba)):
            y = 6 + i * 26
            self.proba_canvas.create_text(8, y + 9, text=c, anchor="w", font=("Segoe UI", 10, "bold"))
            bw = (w - 180) * p
            self.proba_canvas.create_rectangle(80, y, 80 + bw, y + 18, fill=COLORS[c], width=0)
            self.proba_canvas.create_text(86 + bw, y + 9, text=f"{p:.1%}", anchor="w")

    # -------------------------------------------------------------- Live tab
    def toggle_live(self):
        if self.live_job:
            self.after_cancel(self.live_job)
            self.live_job = None
            self.live_btn.config(text="▶ Start monitoring")
            self.status.set("Live monitor paused.")
            return
        src = self.last_results[FEATURES + (["label"] if "label" in self.last_results else [])] \
            if self.last_results is not None else self._sample_frame()
        self.live_source = src.sample(frac=1.0, random_state=random.randrange(1 << 30)).reset_index(drop=True)
        self.live_pos = 0
        self.live_btn.config(text="⏸ Stop monitoring")
        self.status.set("Live monitor running...")
        self._live_tick()

    def _live_tick(self):
        rate = max(1, min(100, int(self.live_speed.get() or 10)))
        batch = max(1, rate // 10)
        if self.live_pos >= len(self.live_source):
            self.live_pos = 0
        chunk = self.live_source.iloc[self.live_pos:self.live_pos + batch]
        self.live_pos += batch
        res = self.detector.detect(chunk, self.threshold.get())
        for (_, r), (_, p) in zip(chunk.iterrows(), res.iterrows()):
            self.live_stats["Processed"] += 1
            fam = p["prediction"]
            truth = to_family(pd.Series([r["label"]]))[0] if "label" in chunk else None
            if fam != "Normal":
                self.live_stats["Alerts"] += 1
                self.live_stats[fam] += 1
            if truth is not None and (truth != "Normal") == (fam != "Normal"):
                self.live_stats["Correct"] += 1
            n = self.live_stats["Processed"]
            tag = "ALERT " if fam != "Normal" else "ok    "
            truth_txt = f" | truth: {r['label']}" if truth is not None else ""
            self.live_log.insert("end", (
                f"#{n:06d} {tag} {fam:<7} p={p['attack_probability']:.2f} "
                f"{r['protocol_type']:<5}{r['service']:<12}{r['flag']:<7}"
                f"src={int(r['src_bytes']):<9}dst={int(r['dst_bytes']):<9}{truth_txt}\n"), fam)
        if int(self.live_log.index("end-1c").split(".")[0]) > 2000:
            self.live_log.delete("1.0", "500.0")
        self.live_log.see("end")
        for key, lbl in self.live_labels.items():
            if key == "Correct":
                n = max(self.live_stats["Processed"], 1)
                lbl.config(text=f"{self.live_stats['Correct'] / n:.0%}")
            else:
                lbl.config(text=f"{self.live_stats[key]:,}")
        delay = int(1000 * batch / rate)
        self.live_job = self.after(delay, self._live_tick)

    def clear_live(self):
        self.live_stats.clear()
        self.live_log.delete("1.0", "end")
        for lbl in self.live_labels.values():
            lbl.config(text="0")


def self_test(report_path: str) -> int:
    """Headless check of the packaged build: IDS.exe --selftest report.txt"""
    try:
        bundle = joblib.load(resource_dir() / "models" / MODEL_FILENAME)
        det = Detector(bundle)
        df = load_records(resource_dir() / "samples" / "sample_traffic.csv")
        res = det.detect(df, bundle.get("default_threshold", 0.3))
        truth = to_family(df["label"]) != "Normal"
        flag = res["alert"] == "ALERT"
        text = (f"SELFTEST OK\nrows={len(df)}\nalerts={int(flag.sum())}\n"
                f"accuracy={(truth.values == flag.values).mean():.4f}\n"
                f"predictions={res['prediction'].value_counts().to_dict()}\n")
        code = 0
    except Exception as exc:  # noqa: BLE001
        text, code = f"SELFTEST FAILED\n{type(exc).__name__}: {exc}\n", 1
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return code


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "--selftest":
        sys.exit(self_test(sys.argv[2]))
    app = IDSApp()
    if app.ok:
        app.mainloop()
