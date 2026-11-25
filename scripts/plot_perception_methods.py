"""Generate the two-column methods figure for the report and the video."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]


def main(out):
    fig, ax = plt.subplots(figsize=(7.15, 3.65))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis('off')
    fig.subplots_adjust(0, 0, 1, 1)
    ink = '#183b3f'
    def box(x, y, w, h, lines, face):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.005,rounding_size=0.012',
                                    facecolor=face, edgecolor='#93a7a8', lw=.75))
        ax.text(x+w/2, y+h/2, lines, ha='center', va='center', size=9.3, color=ink)

    def arrow(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='-|>', lw=.95, color=ink,
                                    shrinkA=0, shrinkB=2, mutation_scale=10))

    ax.text(.018,.945,'Training on nominal reaches', size=12, weight='bold', color=ink)
    box(.02,.715,.255,.155,'Delayed true palm pose\nand velocity','#f2eee5')
    box(.354,.715,.245,.155,'Privileged teacher\ntrained by imitation','#f2eee5')
    box(.68,.715,.30,.155,'Train depth estimator with\npose and action supervision','#dcefed')
    arrow(.282,.792,.345,.792)
    arrow(.607,.792,.669,.792)
    ax.plot([.018,.982],[.65,.65], color='#becbcb', lw=.8)
    ax.text(.018,.566,'Frozen controllers on paired, unseen reaches', size=12, weight='bold', color=ink)
    box(.02,.335,.245,.15,'Depth camera +\nour arm state','#eef3f4')
    box(.357,.335,.25,.15,'Learned palm estimate\n+ frozen teacher actor','#dcefed')
    box(.69,.335,.29,.15,'Joint velocity command\n20 Hz','#dcefed')
    arrow(.272,.41,.349,.41)
    arrow(.614,.41,.682,.41)
    box(.02,.095,.245,.15,'Depth + arm state\n+ known static layout','#eef3f4')
    box(.357,.095,.25,.15,'Geometric tracker\n+ differential IK servo','#f4e7d9')
    box(.69,.095,.29,.15,'Joint position command\n100 Hz','#f4e7d9')
    arrow(.272,.17,.349,.17)
    arrow(.614,.17,.682,.17)
    ax.text(.02,.025,'Each controller moves its own arm. Ground-truth geometry scores the first event.',
            size=8.5, color=ink)
    out.mkdir(parents=True, exist_ok=True)
    for fmt in ('png','pdf'):
        fig.savefig(out / f'methods.{fmt}', dpi=240, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'docs/presentation_assets')
    main(p.parse_args().out)
