#!/usr/bin/env python3
"""Render inspection plots from released aggregate CSVs only; no model evaluation."""
from __future__ import annotations
import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data-dir',type=Path,default=ROOT/'results'/'manuscript_figure_data'); p.add_argument('--output-dir',type=Path,default=ROOT/'results'/'released_plots'); return p.parse_args()
def main():
 a=args(); d=a.data_dir; o=a.output_dir; o.mkdir(parents=True,exist_ok=True)
 # Capacity: retain distinct published conditions and their reported spread/intervals.
 x=pd.read_csv(d/'fig02_capacity.csv'); fig,ax=plt.subplots(figsize=(7,4));
 for (panel,condition),g in x[x.row_type.isin(['seed_summary','subject_summary'])].groupby(['panel','condition']):
  g=g.sort_values('parameters')
  if g.sd_cc.notna().all(): yerr=g.sd_cc
  else: yerr=(g.mean_cc-g.ci95_low, g.ci95_high-g.mean_cc)
  ax.errorbar(g.parameters,g.mean_cc,yerr=yerr,fmt='o-',capsize=2,label=f'{panel}: {condition}')
 ax.set_xscale('log');ax.set(xlabel='Parameters',ylabel='Correlation coefficient');ax.legend();fig.tight_layout();fig.savefig(o/'fig02_capacity.png',dpi=200);plt.close(fig)
 # IV-2a: all stored subject effects.
 x=pd.read_csv(d/'fig03_iv2a_subject_effects.csv'); fig,ax=plt.subplots(figsize=(8,4));
 for key,g in x[x.row_type.eq('subject')].groupby(['classifier_label','recipe']): ax.scatter([f'{key[0]}\n{key[1]}']*len(g),g.delta_accuracy,s=12)
 ax.axhline(0,color='0.4');ax.tick_params(axis='x',rotation=65);ax.set_ylabel('Denoised minus noisy accuracy');fig.tight_layout();fig.savefig(o/'fig03_iv2a.png',dpi=200);plt.close(fig)
 # Metric-utility display values.
 x=pd.read_csv(d/'fig04_centered_display.csv'); fig,ax=plt.subplots(figsize=(6,4));
 for key,g in x.groupby(['classifier','recipe']):ax.scatter(g.centered_sdr_db,g.centered_delta_accuracy,s=10,label='/'.join(key))
 ax.axhline(0,color='0.5');ax.axvline(0,color='0.5');ax.set(xlabel='Centered SDR (dB)',ylabel='Centered accuracy change');ax.legend(fontsize=6,ncol=2);fig.tight_layout();fig.savefig(o/'fig04_metric_utility.png',dpi=200);plt.close(fig)
 # Sleep: released subject-level deltas.
 x=pd.read_csv(d/'fig05_sleepedf_subject_effects.csv'); x=x[x.row_type.eq('subject')]; fig,ax=plt.subplots(figsize=(8,4));
 for key,g in x.groupby(['condition','denoiser_label']):ax.scatter([f'{key[0]}\n{key[1]}']*len(g),g.delta_balanced_accuracy,s=5)
 ax.axhline(0,color='0.4');ax.tick_params(axis='x',rotation=65);ax.set_ylabel('Balanced-accuracy change');fig.tight_layout();fig.savefig(o/'fig05_sleepedf.png',dpi=200);plt.close(fig)
if __name__=='__main__': main()
