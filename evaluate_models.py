"""
Krishika AI — Full Model Evaluation
=====================================
Metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, Classification Report
Saves results to: evaluation_results/
"""

import os
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(BASE_DIR, 'output')
EVAL_DIR    = os.path.join(BASE_DIR, 'evaluation_results')
os.makedirs(EVAL_DIR, exist_ok=True)

# ── Colour palette ─────────────────────────────────────────────────────────
DARK_BG  = '#050d1a'
CARD_BG  = '#0b1829'
CARD_BG2 = '#0f2035'
GREEN    = '#00e87a'
TEAL     = '#00c9a7'
BLUE     = '#3b82f6'
YELLOW   = '#f59e0b'
RED      = '#ef4444'
TEXT     = '#f0f6ff'
SUBTEXT  = '#8ba3bf'

plt.rcParams.update({
    'font.family'   : 'DejaVu Sans',
    'figure.facecolor': DARK_BG,
    'axes.facecolor': CARD_BG,
    'axes.edgecolor': '#1e3a55',
    'axes.labelcolor': TEXT,
    'xtick.color'   : SUBTEXT,
    'ytick.color'   : SUBTEXT,
    'text.color'    : TEXT,
    'grid.color'    : '#1e3a55',
    'grid.linestyle': '--',
    'grid.alpha'    : 0.5,
})

def bar_color(acc):
    if acc >= 0.98: return GREEN
    if acc >= 0.95: return '#4ade80'
    if acc >= 0.90: return BLUE
    if acc >= 0.75: return YELLOW
    return RED

# ════════════════════════════════════════════════════════════════════════════
# EVALUATION LOOP
# ════════════════════════════════════════════════════════════════════════════
crop_folders = sorted(os.listdir(OUTPUT_ROOT))
all_results  = []

print(f"\n{'='*65}")
print("  KRISHIKA AI - MODEL EVALUATION")
print(f"{'='*65}\n")

for folder in crop_folders:
    folder_path = os.path.join(OUTPUT_ROOT, folder)
    if not os.path.isdir(folder_path):
        continue

    files   = os.listdir(folder_path)
    feat_f  = next((f for f in files if f.endswith('_features.npy')), None)
    lab_f   = next((f for f in files if f.endswith('_labels.npy')),   None)
    model_f = next((f for f in files if f.endswith('_model.pkl')),    None)

    if not (feat_f and lab_f and model_f):
        print(f"  Skipping {folder} - missing files")
        continue

    print(f"{'─'*65}")
    print(f"  {folder.upper()}")
    print(f"{'─'*65}")

    X     = np.load(os.path.join(folder_path, feat_f))
    y     = np.load(os.path.join(folder_path, lab_f))
    model = joblib.load(os.path.join(folder_path, model_f))

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test,   y_pred, average='weighted', zero_division=0)
    f1   = f1_score(y_test,       y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    cr   = classification_report(y_test, y_pred, zero_division=0)

    n_classes = len(np.unique(y))
    try:
        class_names = [str(c) for c in model.classes_]
    except Exception:
        class_names = [f"Class {i}" for i in range(n_classes)]

    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  Classes   : {n_classes}  |  Test samples: {len(y_test)}")
    print()
    print("  Classification Report:")
    for line in cr.strip().split('\n'):
        print("    " + line)
    print()

    all_results.append({
        'crop'       : folder,
        'accuracy'   : acc,
        'precision'  : prec,
        'recall'     : rec,
        'f1'         : f1,
        'n_classes'  : n_classes,
        'n_test'     : len(y_test),
        'cm'         : cm,
        'class_names': class_names,
        'report'     : cr,
    })

    # ── Per-crop confusion matrix ──────────────────────────────────────────
    fig_h = max(5, n_classes * 0.7 + 2)
    fig_w = max(6, n_classes * 0.9 + 2)
    fig_cm, ax_cm = plt.subplots(figsize=(fig_w, fig_h))
    fig_cm.patch.set_facecolor(DARK_BG)
    ax_cm.set_facecolor(CARD_BG)

    short_names = []
    for cn in class_names:
        words = cn.replace('_', ' ').split()
        short_names.append('\n'.join(words[:2]) if len(words) > 2
                           else cn.replace('_', '\n'))

    cmap = sns.color_palette("blend:#0b1829,#00e87a", as_cmap=True)
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap,
        xticklabels=short_names, yticklabels=short_names,
        ax=ax_cm, linewidths=0.5, linecolor='#1e3a55',
        annot_kws={'size': max(7, 12 - n_classes), 'color': TEXT},
        cbar_kws={'shrink': 0.8}
    )
    ax_cm.set_title(f'{folder} - Confusion Matrix  (Acc: {acc*100:.2f}%)',
                    fontsize=13, fontweight='bold', color=GREEN, pad=14)
    ax_cm.set_xlabel('Predicted', fontsize=11, color=SUBTEXT)
    ax_cm.set_ylabel('Actual',    fontsize=11, color=SUBTEXT)
    plt.xticks(rotation=40, ha='right', fontsize=8)
    plt.yticks(rotation=0,  fontsize=8)
    ax_cm.collections[0].colorbar.ax.yaxis.set_tick_params(color=SUBTEXT)
    plt.tight_layout()
    cm_path = os.path.join(EVAL_DIR, f'{folder}_confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight', facecolor=DARK_BG)
    plt.close()
    print(f"  Saved confusion matrix -> {os.path.basename(cm_path)}\n")

# ════════════════════════════════════════════════════════════════════════════
# SUMMARY DASHBOARD  (3 panels, no overlapping)
# ════════════════════════════════════════════════════════════════════════════
if not all_results:
    print("No results to plot.")
    exit()

all_results.sort(key=lambda x: x['accuracy'], reverse=True)
crops  = [r['crop']     for r in all_results]
accs   = [r['accuracy'] for r in all_results]
precs  = [r['precision']for r in all_results]
recs   = [r['recall']   for r in all_results]
f1s    = [r['f1']       for r in all_results]
colors = [bar_color(a)  for a in accs]
n      = len(all_results)

# ── Canvas ────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(22, 30), facecolor=DARK_BG)

# top=0.91  →  9% reserved above subplots for the title text
# bottom=0.03, hspace=0.55  →  generous breathing room between panels
gs = gridspec.GridSpec(
    3, 1,
    figure = fig,
    top    = 0.91,
    bottom = 0.03,
    left   = 0.11,
    right  = 0.96,
    hspace = 0.60,
)

# ── Title (drawn in figure coords, safely above gs top=0.91) ─────────────
fig.text(0.5, 0.976,
         'Krishika AI  -  Model Evaluation Dashboard',
         ha='center', va='top',
         fontsize=26, fontweight='bold', color=GREEN)
fig.text(0.5, 0.959,
         'VGG19 Feature Extraction  +  Per-Crop SVM'
         '   |   Accuracy  .  Precision  .  Recall  .  F1   |   17 Crops',
         ha='center', va='top', fontsize=13, color=SUBTEXT)

def styled_ax(ax, title, xlabel='', ylabel=''):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, fontsize=16, fontweight='bold', color=TEXT, pad=14)
    ax.set_xlabel(xlabel, fontsize=13, color=SUBTEXT, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=13, color=SUBTEXT, labelpad=8)
    for spine in ax.spines.values():
        spine.set_color('#1e3a55')
        spine.set_linewidth(0.8)

# ────────────────────────────────────────────────────────────────────────────
# PANEL 1 — Accuracy bar chart
# ────────────────────────────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])

bars = ax1.barh(crops[::-1], [a * 100 for a in accs[::-1]],
                color=colors[::-1], height=0.60, edgecolor='none')

for bar, val in zip(bars, accs[::-1]):
    ax1.text(val * 100 + 0.4,
             bar.get_y() + bar.get_height() / 2,
             f'{val * 100:.2f}%',
             va='center', fontsize=11, fontweight='bold', color=TEXT)

ax1.set_xlim(0, 113)
for xv in [90, 95, 98]:
    ax1.axvline(xv, color='#1e3a55', linestyle='--', linewidth=1)
ax1.grid(axis='x', color='#1e3a55', linestyle='--', alpha=0.4)
ax1.tick_params(axis='y', labelsize=12, colors=TEXT)
ax1.tick_params(axis='x', labelsize=11, colors=SUBTEXT)
styled_ax(ax1, 'Model Accuracy Across Crop Types', 'Accuracy (%)')

legend_patches = [
    Patch(color=GREEN,     label='S-Tier  >= 98%'),
    Patch(color='#4ade80', label='A-Tier  95 - 97%'),
    Patch(color=BLUE,      label='B-Tier  90 - 94%'),
    Patch(color=YELLOW,    label='C-Tier  75 - 89%'),
    Patch(color=RED,       label='D-Tier  < 75%'),
]
ax1.legend(handles=legend_patches, loc='lower right',
           facecolor=CARD_BG2, edgecolor='#1e3a55',
           labelcolor=TEXT, fontsize=11, framealpha=0.9)

# ────────────────────────────────────────────────────────────────────────────
# PANEL 2 — Precision / Recall / F1
# ────────────────────────────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[1])

x = np.arange(n)
w = 0.25
b1 = ax2.bar(x - w, [p * 100 for p in precs], w,
             label='Precision', color=BLUE,  alpha=0.92, edgecolor='none')
b2 = ax2.bar(x,     [r * 100 for r in recs],  w,
             label='Recall',    color=TEAL,  alpha=0.92, edgecolor='none')
b3 = ax2.bar(x + w, [f * 100 for f in f1s],   w,
             label='F1-Score',  color=GREEN, alpha=0.92, edgecolor='none')

ax2.set_xticks(x)
ax2.set_xticklabels(crops, rotation=38, ha='right', fontsize=11)
ax2.tick_params(axis='y', labelsize=11, colors=SUBTEXT)
ax2.set_ylim(0, 120)
ax2.grid(axis='y', color='#1e3a55', linestyle='--', alpha=0.4)
ax2.grid(axis='x', visible=False)
ax2.legend(loc='lower right', facecolor=CARD_BG2, edgecolor='#1e3a55',
           labelcolor=TEXT, fontsize=12, framealpha=0.9)
styled_ax(ax2, 'Precision, Recall and F1-Score by Crop', '', 'Score (%)')

for bars_grp, vals in [(b1, precs), (b2, recs), (b3, f1s)]:
    for bar, val in zip(bars_grp, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f'{val * 100:.2f}',
                 ha='center', va='bottom', fontsize=7, color=SUBTEXT)

# ────────────────────────────────────────────────────────────────────────────
# PANEL 3 — Summary table
# ────────────────────────────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
ax3.axis('off')
ax3.set_facecolor(DARK_BG)

col_labels = ['Crop', 'Accuracy', 'Precision', 'Recall', 'F1-Score',
              'Classes', 'Test Samples']
table_data = [
    [r['crop'],
     f"{r['accuracy']  * 100:.2f}%",
     f"{r['precision'] * 100:.2f}%",
     f"{r['recall']    * 100:.2f}%",
     f"{r['f1']        * 100:.2f}%",
     str(r['n_classes']),
     str(r['n_test'])]
    for r in all_results
]

tbl = ax3.table(
    cellText  = table_data,
    colLabels = col_labels,
    cellLoc   = 'center',
    loc       = 'center',
    bbox      = [0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('#1e3a55')
    cell.set_linewidth(0.7)
    cell.set_height(0.062)
    if row == 0:
        cell.set_facecolor('#0f2035')
        cell.set_text_props(color=GREEN, fontweight='bold', fontsize=13)
    else:
        cell.set_facecolor(CARD_BG if row % 2 == 0 else DARK_BG)
        if col == 1:
            acc_val = all_results[row - 1]['accuracy']
            cell.set_text_props(color=bar_color(acc_val),
                                fontweight='bold', fontsize=12)
        else:
            cell.set_text_props(color=TEXT, fontsize=12)

ax3.set_title('Full Evaluation Summary', fontsize=16, fontweight='bold',
              color=TEXT, pad=16)

# ── Save dashboard ─────────────────────────────────────────────────────────
dash_path = os.path.join(EVAL_DIR, 'evaluation_dashboard.png')
plt.savefig(dash_path, dpi=160, bbox_inches='tight', facecolor=DARK_BG)
plt.close()
print(f"\n{'='*65}")
print(f"  EVALUATION DASHBOARD SAVED")
print(f"  {dash_path}")
print(f"{'='*65}")

# ── JSON ──────────────────────────────────────────────────────────────────
json_out = [{k: v for k, v in r.items() if k not in ('cm', 'class_names', 'report')}
            for r in all_results]
for i, r in enumerate(all_results):
    json_out[i]['accuracy']  = round(r['accuracy']  * 100, 2)
    json_out[i]['precision'] = round(r['precision'] * 100, 2)
    json_out[i]['recall']    = round(r['recall']    * 100, 2)
    json_out[i]['f1']        = round(r['f1']        * 100, 2)

json_path = os.path.join(EVAL_DIR, 'evaluation_results.json')
with open(json_path, 'w') as f:
    json.dump(json_out, f, indent=2)

print(f"  JSON  ->  {json_path}")
print(f"\n  All {len(all_results)} crops evaluated successfully!\n")
