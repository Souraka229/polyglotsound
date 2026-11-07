"""Rendering helpers for the artist UI (figures and score tables)."""
from __future__ import annotations

import html

import numpy as np

CRIT_LABELS = {"sens": "Sens", "syllabes": "Syllabes", "temps": "Rythme", "accents": "Accents", "tons": "Tons",
               "phonetique": "Voyelles", "rimes": "Rimes", "culture": "Culture", "fluidite": "Fluidite"}


def timeline_figure(vocals: np.ndarray, sr: int, f0_times, f0_hz, beats, templates):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    mono = vocals.mean(0) if vocals.ndim == 2 else vocals
    step = max(1, len(mono) // 6000)
    t = np.arange(0, len(mono), step) / sr
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5.2), sharex=True, gridspec_kw={"height_ratios": [1, 1.4]})
    fig.patch.set_facecolor("#0f0d16")
    for ax in (ax1, ax2):
        ax.set_facecolor("#16131f")
        ax.tick_params(colors="#b9b3c9", labelsize=8)
        for s in ax.spines.values():
            s.set_color("#2b2638")
    ax1.fill_between(t, mono[::step], -mono[::step], color="#f5a524", alpha=0.75, lw=0)
    for b in beats or []:
        ax1.axvline(b, color="#ffffff", alpha=0.12, lw=0.6)
    ax1.set_ylabel("Voix", color="#b9b3c9", fontsize=9)
    if f0_times is not None and len(f0_times):
        midi = 69 + 12 * np.log2(np.where(np.isfinite(f0_hz), f0_hz, np.nan) / 440.0)
        ax2.plot(f0_times, midi, color="#7c5cff", lw=1.4)
    for tpl in templates or []:
        for syl, slot in zip(tpl.syllables, tpl.slots):
            y = slot.midi if slot.midi is not None else np.nanmedian([s.midi for s in tpl.slots if s.midi] or [60])
            ax2.add_patch(plt.Rectangle((slot.start, y - 0.45), slot.duration, 0.9,
                                        color="#f5a524" if slot.beat_strength >= 0.6 else "#5a5170", alpha=0.85))
            ax2.text(slot.start + slot.duration / 2, y + 0.9, syl.text, ha="center", fontsize=7, color="#ece8f5")
    ax2.set_ylabel("Hauteur (MIDI)", color="#b9b3c9", fontsize=9)
    ax2.set_xlabel("Temps (s)", color="#b9b3c9", fontsize=9)
    fig.tight_layout()
    return fig


def _bar(score):
    if score is None:
        return '<span class="ps-na">non mesurable</span>'
    pct = int(round(100 * score))
    hue = "#3ecf8e" if score >= 0.75 else "#f5a524" if score >= 0.5 else "#ef5b5b"
    return (f'<div class="ps-bar"><div style="width:{pct}%;background:{hue}"></div></div>'
            f'<span class="ps-pct">{pct}</span>')


def adaptation_cards(adaptations: list[dict]) -> str:
    out = []
    for i, a in enumerate(adaptations):
        out.append(f'<section class="ps-line"><div class="ps-orig"><b>Ligne {i + 1}</b> « {html.escape(a["original"])} »'
                   f'<div class="ps-lit">Traduction litterale : {html.escape(a["literal"])}</div></div><div class="ps-cards">')
        for k, c in enumerate(a["top3"]):
            badge = "rejetee" if c["rejected"] else ("meilleure" if k == 0 else f"option {k + 1}")
            total = "?" if c["total"] is None else int(round(100 * c["total"]))
            rows = "".join(
                f'<tr title="{html.escape(cr["explanation"])}"><td>{CRIT_LABELS.get(cr["name"], cr["name"])}'
                f'{"" if cr["reliable"] else " <i>*</i>"}</td><td>{_bar(cr["score"])}</td></tr>'
                for cr in c["criteria"])
            reasons = "".join(f"<li>{html.escape(r)}</li>" for r in c["rejection_reasons"])
            syl = " · ".join(html.escape(s["text"]) for s in c["syllables"])
            out.append(f'<article class="ps-card {"ps-rej" if c["rejected"] else ""}">'
                       f'<header><span class="ps-badge">{badge}</span><span class="ps-total">{total}</span></header>'
                       f'<p class="ps-text">{html.escape(c["text"])}</p><p class="ps-syl">{syl}</p>'
                       f'<table>{rows}</table>{"<ul class=ps-reasons>" + reasons + "</ul>" if reasons else ""}'
                       f'<footer>source : {html.escape(c["source"])}</footer></article>')
        out.append(f'</div><details><summary>Pourquoi ce classement ?</summary><pre>{html.escape(a["explanation"])}</pre>'
                   f'</details></section>')
    out.append('<p class="ps-foot">* critere heuristique : a confirmer a l\'ecoute ou par un locuteur natif. '
               'Survolez une ligne pour lire l\'explication.</p>')
    return "".join(out)


CSS = """
/* Colors follow the Gradio theme variables: readable in light AND dark mode. */
.gradio-container{max-width:1180px!important;margin:auto}
#ps-hero{padding:18px 20px;border-radius:16px;margin-bottom:8px;
  background:linear-gradient(120deg,#f5a524 0%,#e8590c 55%,#7c3aed 100%);color:#fff}
#ps-hero h1{font-size:2.1rem;letter-spacing:-.02em;margin:0;color:#fff}
#ps-hero p{margin:.3rem 0 0;color:#fff;opacity:.95;font-size:1.02rem}
#ps-steps{font-size:1rem}
.ps-line{border:1px solid var(--border-color-primary);border-radius:14px;padding:14px;margin:12px 0;
  background:var(--block-background-fill);color:var(--body-text-color)}
.ps-orig{font-size:1.05rem}.ps-lit{color:var(--body-text-color-subdued);font-size:.92rem;margin-top:4px}
.ps-cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:10px;margin-top:10px}
.ps-card{border:1px solid var(--border-color-primary);border-radius:12px;padding:12px;
  background:var(--background-fill-secondary);color:var(--body-text-color)}
.ps-card header{display:flex;justify-content:space-between;align-items:center}
.ps-badge{font-size:.75rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700;color:#c2410c}
.ps-rej .ps-badge{color:#dc2626}.ps-rej{opacity:.75}
.ps-total{font-size:1.7rem;font-weight:800}
.ps-text{font-size:1.1rem;font-weight:600;margin:.4rem 0}
.ps-syl{color:var(--body-text-color-subdued);font-size:.85rem;margin:0 0 .5rem}
.ps-card table{width:100%;font-size:.88rem;border-collapse:collapse}.ps-card td{padding:3px 0}
.ps-bar{display:inline-block;width:68%;height:8px;background:var(--border-color-primary);border-radius:4px;vertical-align:middle}
.ps-bar div{height:8px;border-radius:4px}.ps-pct{margin-left:6px;font-weight:600}
.ps-na{color:var(--body-text-color-subdued);font-style:italic}
.ps-reasons{color:#dc2626;font-size:.85rem;margin:.4rem 0 0 1rem}
.ps-card footer{color:var(--body-text-color-subdued);font-size:.78rem;margin-top:6px}
.ps-foot{color:var(--body-text-color-subdued);font-size:.85rem}
details pre{white-space:pre-wrap;font-size:.85rem}
.ps-big button{font-size:1.1rem!important;padding:14px!important}
"""
