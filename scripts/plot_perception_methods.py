"""Generate the methods figure and the source slide for the video."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]


def methods_figure(out):
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


def references_slide(out):
    entries = [
        ('Equipping the Baxter Robot with\nHuman-Inspired Hand-Clapping Skills',
         'Fitter & Kuchenbecker · IEEE RO-MAN, 2016', 'doi.org/10.1109/ROMAN.2016.7745097'),
        ('Visual Servo Control II.\nAdvanced Approaches',
         'Chaumette & Hutchinson · IEEE RAM, 2007', 'doi.org/10.1109/MRA.2007.339609'),
        ('Learning by Cheating',
         'Chen et al. · CoRL, PMLR 100, 2020', 'proceedings.mlr.press/v100/chen20a.html'),
        ('A Reduction of Imitation Learning and Structured\nPrediction to No-Regret Online Learning',
         'Ross, Gordon & Bagnell · AISTATS, 2011', 'proceedings.mlr.press/v15/ross11a.html'),
        ('Agile Catching with Whole-Body MPC\nand Blackbox Policy Learning',
         'Abeyruwan et al. · L4DC, 2023', 'proceedings.mlr.press/v211/abeyruwan23a.html'),
        ('Contemporary Dap Up',
         'OMFGrant · Wikimedia Commons, 2025 · CC0', 'commons.wikimedia.org/wiki/File:Contemporary_Dap_Up.gif'),
    ]
    figure = plt.figure(figsize=(16, 9), facecolor='#152426')
    figure.text(.045, .875, 'References', color='white', size=40, weight='bold')
    for i, (title, credit, url) in enumerate(entries):
        x, y = .045 + .49 * (i % 2), .765 - .254 * (i // 2)
        figure.text(x, y, title, va='top', color='#fafafa', size=18.5)
        figure.text(x, y - .097, credit, va='top', color='#bad4d0', size=15)
        figure.text(x, y - .143, url, va='top', color='#c5cbca', size=12,
                    url='https://' + url)
    for fmt in ('png', 'pdf'):
        figure.savefig(out / f'references_slide.{fmt}', dpi=120, facecolor=figure.get_facecolor())
    plt.close(figure)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=ROOT / 'docs/presentation_assets')
    output = p.parse_args().out
    methods_figure(output)
    references_slide(output)
