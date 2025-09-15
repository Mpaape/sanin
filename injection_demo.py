# demo_plot_all.py
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sanin import AnomalyInjector, AnomalyType

# ---------- 1) Série base ----------
def make_base_series(n=1000, seed=123, period=50):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    level = 100.0
    trend = 0.01 * t
    season = 5.0 * np.sin(2 * np.pi * t / period)
    noise = rng.normal(0, 1.0, n)
    values = level + trend + season + noise
    idx = pd.date_range("2025-01-01", periods=n, freq="min")
    return pd.Series(values, index=idx, name="value")

# ---------- 2) Aux: plot ----------
def plot_before_after(orig: pd.Series, inj: pd.Series, title: str, outdir: str):
    plt.figure(figsize=(12, 4))
    plt.plot(orig.index, orig.values, label="Original")
    plt.plot(inj.index, inj.values, label="Com anomalia")
    plt.title(title)
    plt.xlabel("Tempo")
    plt.ylabel("Valor")
    plt.grid(True)
    plt.legend()
    os.makedirs(outdir, exist_ok=True)
    safe = title.lower().replace(" ", "_").replace("/", "_")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{safe}.png"), dpi=120)
    plt.show()

# ---------- 3) Casos de uso ----------
def run_all():
    s = make_base_series()
    inj = AnomalyInjector(random_state=42)
    outdir = "./figs"

    cases = [
        # (tipo, kwargs, titulo)
        (AnomalyType.SPIKE, dict(severity=1.5, n_points=5), "Spike (5 pontos)"),
        (AnomalyType.DROP, dict(severity=1.5, n_points=5), "Drop (5 pontos)"),
        (AnomalyType.LEVEL_SHIFT, dict(severity=2.0), "Level shift (degrau)"),
        (AnomalyType.VARIANCE_CHANGE, dict(severity=2.0), "Mudança de variância (janela)"),
        (AnomalyType.TREND_DRIFT, dict(severity=1.5), "Trend drift (mudança de inclinação)"),
        (AnomalyType.SEASON_AMP_CHANGE, dict(severity=1.0), "Mudança de amplitude sazonal (janela)"),
        (AnomalyType.FLATLINE, {}, "Flatline (sensor preso)"),
        (AnomalyType.MISSING, {}, "Missing (NaNs em janela)"),
        (AnomalyType.STUCK_HIGH, {}, "Stuck high (saturação alta)"),
        (AnomalyType.STUCK_LOW, {}, "Stuck low (saturação baixa)"),
        (AnomalyType.BLACKOUT, {}, "Blackout (zerar/valor fixo)"),
    ]

    for kind, kwargs, title in cases:
        y, rep, _ = inj.inject(s, kind, return_mask=True, **kwargs)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] {kind.value}: params={rep.params}  n_indices={len(rep.indices)}")
        plot_before_after(s, y, f"{title} - {kind.value}", outdir)

if __name__ == "__main__":
    run_all()
