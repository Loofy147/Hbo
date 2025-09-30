"""Improved GridAnalysisCallback: robust path handling, run on save/train_end, optional wandb/CSV push."""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from transformers import TrainerCallback

plt.rcParams.update({'figure.figsize': (8,5)})

class GridAnalysisCallback(TrainerCallback):
    def __init__(self, raw_csv='grid_raw_results.csv', summary_csv='grid_summary.csv',
                 run_on='train_end', push_to_wandb=False):
        self.raw_csv = raw_csv
        self.summary_csv = summary_csv
        assert run_on in ('train_end', 'on_save')
        self.run_on = run_on
        self.push_to_wandb = push_to_wandb

    def _get_path(self, args, fname):
        if os.path.isabs(fname) or os.path.exists(fname):
            return fname
        return os.path.join(args.output_dir, fname)

    def _run_analysis(self, args):
        RAW_CSV = self._get_path(args, self.raw_csv)
        SUMMARY_CSV = self._get_path(args, self.summary_csv)
        out_dir = args.output_dir
        os.makedirs(out_dir, exist_ok=True)

        if not os.path.exists(RAW_CSV):
            print(f"[GridAnalysisCallback] {RAW_CSV} not found, skipping")
            return

        raw = pd.read_csv(RAW_CSV)
        if os.path.exists(SUMMARY_CSV):
            summary = pd.read_csv(SUMMARY_CSV)
        else:
            summary = raw.groupby(['cs', 'neumann_steps', 'echo_freq', 'lr']).agg(
                mean_val_acc=('final_val_acc', 'mean'),
                std_val_acc=('final_val_acc', 'std'),
                mean_train_loss=('final_train_loss', 'mean'),
                std_train_loss=('final_train_loss', 'std'),
                n_runs=('final_val_acc', 'count'),
                avg_time_s=('time_s', 'mean')
            ).reset_index()

        summary_sorted = summary.sort_values(['cs', 'neumann_steps', 'echo_freq', 'lr'])
        xlabels = summary_sorted.apply(lambda r: f"cs={r.cs}\nN={int(r.neumann_steps)} f={int(r.echo_freq)}", axis=1)
        means = summary_sorted['mean_val_acc']
        stds = summary_sorted['std_val_acc'].fillna(0.0)

        plt.figure()
        plt.bar(range(len(means)), means, yerr=stds, capsize=4)
        plt.xticks(range(len(means)), xlabels, rotation=45, ha='right')
        plt.ylabel('Mean val accuracy')
        plt.title('Mean val accuracy per config (errorbars = std)')
        plt.tight_layout()
        img1 = os.path.join(out_dir, 'mean_val_acc_by_config.png')
        plt.savefig(img1, dpi=150)
        plt.close()

        baseline_mask = (summary['cs'] == 0.0)
        baseline_mean = None
        baseline_time = None
        if baseline_mask.sum() > 0:
            baseline_df = summary[baseline_mask].iloc[0]
            baseline_mean = float(baseline_df['mean_val_acc'])
            baseline_time = float(baseline_df['avg_time_s'])

        def config_key(row):
            return (row['cs'], int(row['neumann_steps']), int(row['echo_freq']), float(row['lr']))

        groups = {}
        for _, r in raw.iterrows():
            k = config_key(r)
            groups.setdefault(k, []).append(float(r['final_val_acc']))

        baseline_vals = []
        if baseline_mean is not None:
            baseline_keys = [k for k in groups.keys() if k[0] == 0.0]
            for k in baseline_keys:
                baseline_vals += groups[k]

        rows = []
        for _, s in summary_sorted.iterrows():
            k = (float(s.cs), int(s.neumann_steps), int(s.echo_freq), float(s.lr))
            vals = np.array(groups.get(k, []))
            mean_val = float(s['mean_val_acc'])
            std_val = float(s['std_val_acc']) if not np.isnan(s['std_val_acc']) else 0.0
            avg_time = float(s['avg_time_s'])
            if len(vals) > 0 and len(baseline_vals) > 0:
                tstat, pval = stats.ttest_ind(vals, baseline_vals, equal_var=False)
                pooled_sd = np.sqrt(((vals.std(ddof=0) ** 2) + (np.std(baseline_vals, ddof=0) ** 2)) / 2.0)
                cohen_d = (vals.mean() - np.mean(baseline_vals)) / (pooled_sd + 1e-12)
            else:
                pval = np.nan
                cohen_d = np.nan
            delta = (mean_val - baseline_mean) if baseline_mean is not None else np.nan
            extra_time = (avg_time - baseline_time) if baseline_time is not None else np.nan
            rel_gain_per_sec = (delta / extra_time) if (extra_time is not None and extra_time > 0) else np.nan
            rows.append({
                'cs': s.cs, 'neumann': s.neumann_steps, 'echo_freq': s.echo_freq, 'lr': s.lr,
                'mean_val': mean_val, 'std_val': std_val, 'pval_vs_baseline': pval,
                'cohens_d': cohen_d, 'delta': delta, 'avg_time_s': avg_time,
                'extra_time_s': extra_time, 'gain_per_sec': rel_gain_per_sec
            })

        stats_df = pd.DataFrame(rows)
        stats_df_sorted = stats_df.sort_values('mean_val', ascending=False).reset_index(drop=True)
        out_csv = os.path.join(out_dir, 'grid_stats_detailed.csv')
        stats_df_sorted.to_csv(out_csv, index=False)

        plt.figure()
        plt.scatter(stats_df['avg_time_s'], stats_df['mean_val'])
        for i, row in stats_df.iterrows():
            plt.text(row['avg_time_s'] + 0.5, row['mean_val'], f"cs={row['cs']},N={int(row['neumann'])}", fontsize=8)
        plt.xlabel('avg time per run (s)')
        plt.ylabel('mean val acc')
        plt.title('Mean val acc vs avg runtime (trade-off)')
        plt.tight_layout()
        img2 = os.path.join(out_dir, 'valacc_vs_time.png')
        plt.savefig(img2, dpi=150)
        plt.close()

        print(f"GridAnalysis: saved {out_csv}, {img1}, {img2}")

    def on_train_end(self, args, state, control, **kwargs):
        if self.run_on == 'train_end':
            self._run_analysis(args)

    def on_save(self, args, state, control, **kwargs):
        if self.run_on == 'on_save':
            self._run_analysis(args)