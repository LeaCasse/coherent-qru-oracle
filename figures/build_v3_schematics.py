"""Vector explanations of the V3 service and its workspace contracts."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Ellipse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/figures/v3'
NAVY, TEAL, GOLD, GRAY = '#234f75', '#207c6e', '#b16727', '#657582'
PALE = {NAVY: '#f0f5fa', TEAL: '#eef7f4', GOLD: '#fcf4eb', GRAY: '#f4f6f8'}


def setup():
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10.5,
                        'mathtext.fontset': 'dejavusans', 'pdf.fonttype': 42,
                        'svg.fonttype': 'none', 'text.color': NAVY})


def box(ax, x, y, width, height, text, color=NAVY, size=11, dash=False):
    ax.add_patch(FancyBboxPatch((x-width/2, y-height/2), width, height,
        boxstyle='round,pad=0,rounding_size=.35', ec=color, fc=PALE[color],
        lw=1.15, linestyle=(0, (3, 2)) if dash else '-', zorder=3))
    ax.text(x, y, text, ha='center', va='center', fontsize=size, zorder=4)


def arrow(ax, start, end, color=GRAY, rad=0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=11,
        lw=1.2, color=color, connectionstyle=f'arc3,rad={rad}', zorder=2))


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(OUT/f'{name}.{ext}', bbox_inches='tight', pad_inches=.045,
                    dpi=180, facecolor='white')
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    fig.subplots_adjust(left=.015, right=.985, top=.98, bottom=.02)
    ax.set(xlim=(0, 100), ylim=(0, 54)); ax.axis('off')
    xs = [12, 37, 62, 87]
    for x in xs:
        ax.add_patch(Circle((x, 39.5), 9.5, fc='#f2f6f9', ec='none'))
    for x in [22.5, 47.5, 72.5]:
        arrow(ax, (x, 39.5), (x+4.3, 39.5), NAVY)
    # The preparation retains geometry beyond its scalar readout.
    ax.add_patch(Circle((12, 39.5), 5.4, fill=False, ec=TEAL, lw=1.3))
    ax.add_patch(Ellipse((12, 39.5), 10.8, 3.7, fill=False, ec=TEAL, lw=.9))
    ax.plot([12, 12], [34.1, 44.9], color=TEAL, lw=.7, alpha=.6)
    arrow(ax, (12, 39.5), (15.3, 43.5), TEAL)
    ax.text(12, 51.7, r'$\sum_x c_x|x\rangle$', ha='center', va='center', fontsize=11)
    # A word and a polynomial are two implementations of the same service.
    for i, h in enumerate([1.4, 3.8, 6, 4.7, 2.2]):
        ax.add_patch(FancyBboxPatch((30.8+2.1*i, 35), 1.35, h,
            boxstyle='round,pad=0,rounding_size=.2', fc=TEAL, ec='none'))
    import numpy as np
    x = np.linspace(31, 43, 60)
    ax.plot(x, 43-1.2*np.cos((x-31)*np.pi/12), color=GOLD, lw=1.5)
    ax.text(37, 51.7, 'QPE / GQSP', ha='center', va='center', fontsize=11)
    ax.add_patch(Circle((62, 39.5), 5.4, fill=False, ec=GOLD, lw=1.3))
    ax.add_patch(Ellipse((62, 39.5), 10.8, 3.7, fill=False, ec=GOLD, lw=.9))
    arrow(ax, (62, 39.5), (65.4, 42.6), GOLD)
    arrow(ax, (56.7, 44.4), (66.6, 44.9), GOLD, rad=-.45)
    ax.text(62, 51.7, r'$V_j(s_x)$', ha='center', va='center', fontsize=11)
    for x, y, c in [(84.8, 38.8, TEAL), (89.1, 38.8, TEAL),
                    (84.8, 35, TEAL), (89.1, 35, GOLD)]:
        ax.add_patch(Circle((x, y), 1.15, fc=c, ec='none'))
    arrow(ax, (92.5, 42), (82, 42), NAVY, rad=.6)
    ax.text(87, 51.7, 'Workspace return', ha='center', va='center', fontsize=10.1)
    for x, title, detail in zip(xs, ['Prepare', 'Encode', 'Control', 'Handle work'],
        ['QRU circuit and lift', 'Word or polynomial', 'Retain input and target', 'Keep, renew or phase']):
        ax.text(x, 27.6, title, ha='center', weight='bold', fontsize=11)
        ax.text(x, 23.5, detail, ha='center', fontsize=9.5)
    ax.plot([1, 99], [20.5, 20.5], color='#dce4ea', lw=.8)
    for x, title, detail, c in [
        (16, 'Preparation geometry', 'Fibre covariance\nInvariant initialization', TEAL),
        (50, 'One-call contract', 'Clean action and leakage\nLiteral workspace return', NAVY),
        (84, 'Protocol certificate', 'Interventions and memory\nSelected renewal boundaries', GOLD)]:
        ax.text(x, 16.3, title, ha='center', weight='bold', fontsize=10.4, color=c)
        ax.text(x, 8.5, detail, ha='center', va='center', fontsize=10, linespacing=1.6)
    save(fig, 'v3_framework')


def wire(ax, y, label, start=17, end=96, bundle=False):
    ax.text(1, y, label, ha='left', va='center', fontsize=10.7)
    ax.plot([start, end], [y, y], color='#91a3b1', lw=.9, zorder=0)
    if bundle:
        ax.plot([start+.5, start+1.5], [y-.17, y+.17], color=GRAY, lw=.9)


def gate(ax, x, ys, label, color=NAVY, width=11, dash=False, size=11):
    box(ax, x, (min(ys)+max(ys))/2, width, max(ys)-min(ys)+.86, label, color, size, dash)


def controlled(ax, x, ycontrol, ytarget, label, color=NAVY, width=11, word=False):
    ax.plot([x, x], [ytarget, ycontrol], color=color, lw=1, zorder=1)
    ax.scatter([x], [ycontrol], s=25, marker='s' if word else 'o',
        facecolors='white' if word else color, edgecolors=color, zorder=5)
    gate(ax, x, [ytarget], label, color, width)


def circuits():
    fig, axes = plt.subplots(3, 1, figsize=(7.4, 5.9), gridspec_kw={'height_ratios': [1.25, 1.25, 1]})
    fig.subplots_adjust(left=.02, right=.99, top=.985, bottom=.02, hspace=.26)
    for ax in axes:
        ax.set(xlim=(0, 100), ylim=(0, 8.5)); ax.axis('off')
    ax = axes[0]
    ax.text(1, 8, '(a)  A complete numerical call', weight='bold', fontsize=11)
    for y, label, bundle in [(6.35, '$X$', True), (4.45, r'$W:|0\rangle$', True), (2.55, '$T$', False)]:
        wire(ax, y, label, bundle=bundle)
    gate(ax, 30, [6.35, 4.45], '$E_X$', TEAL, 14)
    controlled(ax, 52, 4.45, 2.55, '$V_j(g_Y)$', GOLD, 19, word=True)
    gate(ax, 74, [6.35, 4.45], r'$E_X^\dagger$', TEAL, 14)
    gate(ax, 92, [4.45], r'$\mathrm{Tr}_W$', GRAY, 12, size=10)
    ax.text(57, .6, 'Keep work between calls, or renew it at a segment boundary', ha='center', fontsize=9.7)
    ax = axes[1]
    ax.text(1, 8.05, '(b)  Two-bit estimator', weight='bold', fontsize=11)
    for y, label, bundle in [(6.5, '$y_0$', False), (5.05, '$y_1$', False), (3.6, '$a$', False), (1.6, '$X,S$', True)]:
        wire(ax, y, label, bundle=bundle)
    gate(ax, 25, [6.5, 3.6], '$P_a$', TEAL, 11)
    gate(ax, 25, [1.6], '$A_X$', GRAY, 11, dash=True)
    for x, y in [(43, 6.5), (59, 5.05), (74, 5.05)]:
        controlled(ax, x, y, 1.6, '$Q_X$', NAVY, 10)
    gate(ax, 91, [6.5, 5.05], r'$\mathsf{F}_4^\dagger$', TEAL, 11)
    ax.text(57, .05, 'Prepared: include the dashed gate    Invariant: omit it', ha='center', fontsize=9.7)
    ax = axes[2]
    ax.text(1, 8, '(c)  Controlled reflections', weight='bold', fontsize=11)
    for y, label, bundle in [(6.05, '$c$', False), (4.2, '$X$', True), (2.1, '$S$', False)]:
        wire(ax, y, label, bundle=bundle)
    controlled(ax, 29, 6.05, 2.1, '$Z$', NAVY, 8)
    gate(ax, 47, [4.2, 2.1], r'$A_X^\dagger$', TEAL, 13)
    controlled(ax, 65, 6.05, 2.1, '$Z$', NAVY, 8)
    gate(ax, 84, [4.2, 2.1], '$A_X$', TEAL, 13)
    ax.text(58, .35, 'The phase bit controls the reflections only', ha='center', fontsize=9.7)
    save(fig, 'v3_circuits')


def main():
    setup()
    overview()
    circuits()


if __name__ == '__main__':
    main()
