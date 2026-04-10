import argparse
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--save", action="store_true", help="Save figure to file")
args = parser.parse_args()

aw = 2
fs = 24
font = {"size": fs}
matplotlib.rc("font", **font)
matplotlib.rc("axes", linewidth=aw)


def set_fig_properties(ax_list):
    tl = 8
    tw = 2
    tlm = 4

    for ax in ax_list:
        ax.tick_params(which="major", length=tl, width=tw)
        ax.tick_params(which="minor", length=tlm, width=tw)
        ax.tick_params(which="both", axis="both", direction="in", right=True, top=True)


with open("omega2.out", "r") as f:
    first_line = f.readline().strip().lstrip("#").split()
    sym_points = first_line[: len(first_line) // 2]
    sym_points = [float(x) for x in sym_points]
omega2_array = np.loadtxt("omega2.out")
linear_path = omega2_array[:, 0]
nu = np.sqrt(omega2_array[:, 1:]) / (2 * np.pi)

plt.figure(figsize=(10, 10))
set_fig_properties([plt.gca()])
plt.plot(linear_path, nu, color="C0", lw=3)
plt.xlim([0, max(linear_path)])
plt.vlines(sym_points, ymin=0, ymax=17)
plt.gca().set_xticks(sym_points)
plt.gca().set_xticklabels([r"$\Gamma$", "X", "K", r"$\Gamma$", "L"])
plt.ylim([0, 17])
plt.ylabel(r"$\nu$ (THz)")
if args.save:
    plt.savefig("phonon_dispersion.png", dpi=300, bbox_inches="tight")
else:
    plt.show()
