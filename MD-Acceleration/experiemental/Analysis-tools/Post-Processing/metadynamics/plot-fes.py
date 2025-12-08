#!/usr/bin/env python3
"""
plot_fes.py

Improved plotting script for PLUMED-style fes.dat.

Features added (per user request):
 - produce each plot in its own figure (3D surface, 2D contour, 1D projections)
 - 2D contour includes contour lines with labels and marks local minima
 - minima are also shown on the 3D surface and on the 1D slices

Usage examples:
  python plot_fes.py fes.dat               # show plots interactively
  python plot_fes.py fes.dat --save --out myfes   # save each plot separately as PNGs
  python plot_fes.py fes.dat --slice-angle 0.5    # also produce a 1D slice at the requested angle

"""

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def parse_fes_file(path):
    header_info = {}
    fields = None
    data_lines = []

    with open(path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith('#'):
                if s.startswith('#!'):
                    tok = s[2:].strip()
                    if tok.upper().startswith('FIELDS'):
                        parts = tok.split()
                        fields = parts[1:]
                    elif tok.upper().startswith('SET'):
                        parts = tok.split()
                        if len(parts) >= 3:
                            name = parts[1]
                            try:
                                val = float(parts[2])
                                if name.startswith('nbins'):
                                    val = int(val)
                            except ValueError:
                                val = parts[2]
                            header_info[name] = val
                continue
            data_lines.append(s)

    if len(data_lines) == 0:
        raise ValueError('No data lines found in file')

    data = np.loadtxt(data_lines)
    if fields is None:
        fields = [f'col{i}' for i in range(data.shape[1])]

    return header_info, fields, data


def grid_from_data(data, header_info):
    nbins_x = header_info.get('nbins_sih') or header_info.get('nbins_x')
    nbins_y = header_info.get('nbins_angle_phi') or header_info.get('nbins_y')

    nrows, ncols = data.shape

    if nbins_x and nbins_y:
        if nbins_x * nbins_y != nrows:
            print('Warning: header nbins mismatch; falling back to inference', file=sys.stderr)
            nbins_x = None
            nbins_y = None

    if nbins_x is None or nbins_y is None:
        # infer nbins_x by counting how many rows share the same y as the first row
        first_y = data[0, 1]
        nbins_x_guess = np.sum(np.isclose(data[:, 1], first_y))
        nbins_x = int(nbins_x_guess)
        if nbins_x == 0:
            raise ValueError('Failed to infer grid: nbins_x==0')
        nbins_y = nrows // nbins_x

    x = data[:nbins_x, 0].copy()
    y = data[::nbins_x, 1].copy()

    reshaped = data.reshape((nbins_y, nbins_x, -1))
    return x, y, reshaped


def find_local_minima(Z):
    """Return list of (iy, ix) indices of local minima in 2D array Z.

    This checks interior points (1..-2) and requires strict < comparison with 8 neighbors.
    It also ensures the global minimum is included.
    """
    ny, nx = Z.shape
    minima = []
    for iy in range(1, ny - 1):
        for ix in range(1, nx - 1):
            val = Z[iy, ix]
            neigh = Z[iy - 1:iy + 2, ix - 1:ix + 2]
            # flatten and remove center
            neigh_flat = np.delete(neigh.flatten(), 4)
            if np.all(val < neigh_flat):
                minima.append((iy, ix))
    if len(minima) == 0:
        # fallback: include global minimum
        arg = np.argmin(Z)
        iy, ix = np.unravel_index(arg, Z.shape)
        minima.append((iy, ix))
    else:
        # ensure global min included
        arg = np.argmin(Z)
        g_iy, g_ix = np.unravel_index(arg, Z.shape)
        if (g_iy, g_ix) not in minima:
            minima.append((g_iy, g_ix))
    return minima


def plot_3d_surface(x, y, Z, zlabel='FES', show=True, out_prefix=None):
    Xg, Yg = np.meshgrid(x, y)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(Xg, Yg, Z, cmap='viridis', rcount=100, ccount=100, linewidth=0, antialiased=True)
    ax.set_xlabel('sih')
    ax.set_ylabel('angle_phi')
    ax.set_zlabel(zlabel)
    fig.colorbar(surf, ax=ax, shrink=0.6)
    ax.set_title('3D free-energy surface')

    # mark minima
    minima = find_local_minima(Z)
    for iy, ix in minima:
        ax.scatter(x[ix], y[iy], Z[iy, ix], color='red', s=40)
        ax.text(x[ix], y[iy], Z[iy, ix],
        f"  ({x[ix]:.3f},{y[iy]:.3f})\n{Z[iy,ix]:.3f}",
        color='red')

    plt.tight_layout()
    if out_prefix:
        fname = f"{out_prefix}_3d.png"
        fig.savefig(fname, dpi=200)
        print('Saved', fname)
        plt.close(fig)
    elif show:
        plt.show()


def plot_contour(x, y, Z, zlabel='FES', show=True, out_prefix=None, annotate_minima=True):
    Xg, Yg = np.meshgrid(x, y)
    fig, ax = plt.subplots(figsize=(7, 6))
    # filled contour
    cf = ax.contourf(Xg, Yg, Z, levels=50, cmap='viridis')
    
    # IMPROVED CONTOUR LINES
    # You can adjust the number of levels, colors, and styles here
    contour_levels = 15  # Increase number of contour lines
    cs = ax.contour(Xg, Yg, Z, levels=contour_levels, 
                   colors='black', linewidths=0.8, linestyles='solid')
    
    # Improved contour labels - larger font, better formatting
    ax.clabel(cs, inline=True, fontsize=9, fmt='%.2f', 
             colors='black')  # You can change color if needed
    
    ax.set_xlabel('sih')
    ax.set_ylabel('angle_phi')
    ax.set_title('2D FES contour')
    fig.colorbar(cf, ax=ax)

    if annotate_minima:
        minima = find_local_minima(Z)
        for iy, ix in minima:
            ax.scatter(x[ix], y[iy], marker='o', color='white', 
                      edgecolor='k', s=60, zorder=5)
            ax.annotate(f'{Z[iy,ix]:.3f}', (x[ix], y[iy]), 
                       textcoords='offset points', xytext=(5,5), 
                       color='white', fontsize=8, weight='bold')

    plt.tight_layout()
    if out_prefix:
        fname = f"{out_prefix}_contour.png"
        fig.savefig(fname, dpi=200)
        print('Saved', fname)
        plt.close(fig)
    elif show:
        plt.show()

def plot_1d_projections(x, y, Z, zlabel='FES', show=True, out_prefix=None, slice_angle=None):
    # projections over angle (y) -> as function of x
    mean_over_angle = np.mean(Z, axis=0)
    min_over_angle = np.min(Z, axis=0)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, mean_over_angle, label='mean over angle')
    ax.plot(x, min_over_angle, label='min over angle')
    ax.set_xlabel('sih')
    ax.set_ylabel(zlabel)
    ax.set_title('1D projections (over angle)')
    ax.legend()
    plt.tight_layout()
    if out_prefix:
        fname = f"{out_prefix}_proj_sih.png"
        fig.savefig(fname, dpi=200)
        print('Saved', fname)
        plt.close(fig)
    elif show:
        plt.show()

    # projections over sih (x) -> as function of y
    mean_over_x = np.mean(Z, axis=1)
    min_over_x = np.min(Z, axis=1)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(y, mean_over_x, label='mean over sih')
    ax.plot(y, min_over_x, label='min over sih')
    ax.set_xlabel('angle_phi')
    ax.set_ylabel(zlabel)
    ax.set_title('1D projections (over sih)')
    ax.legend()
    plt.tight_layout()
    if out_prefix:
        fname = f"{out_prefix}_proj_angle.png"
        fig.savefig(fname, dpi=200)
        print('Saved', fname)
        plt.close(fig)
    elif show:
        plt.show()

    # optional exact slice at nearest angle
    if slice_angle is not None:
        idx = int(np.argmin(np.abs(y - slice_angle)))
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(x, Z[idx, :], label=f'slice angle={y[idx]:.6f}')
        ax.set_xlabel('sih')
        ax.set_ylabel(zlabel)
        ax.set_title(f'1D slice at angle_phi = {y[idx]:.6f} (requested {slice_angle})')
        # mark minima on the slice
        min_ix = np.argmin(Z[idx, :])
        ax.scatter(x[min_ix], Z[idx, min_ix], color='red', s=50)
        ax.annotate(f'{Z[idx,min_ix]:.3f}', (x[min_ix], Z[idx, min_ix]), xytext=(5,5), textcoords='offset points')
        ax.legend()
        plt.tight_layout()
        if out_prefix:
            fname = f"{out_prefix}_slice_angle_{idx}.png"
            fig.savefig(fname, dpi=200)
            print('Saved', fname)
            plt.close(fig)
        elif show:
            plt.show()


def main():
    p = argparse.ArgumentParser(description='Plot FES from fes.dat (PLUMED-style)')
    p.add_argument('file', help='path to fes.dat')
    p.add_argument('--zfield', help='name of the field to plot (from FIELDS header). Default: first non-coordinate field', default=None)
    p.add_argument('--save', action='store_true', help='save figures instead of showing them')
    p.add_argument('--out', help='output prefix for saved images', default='fes')
    p.add_argument('--slice-angle', help='plot a 1D slice at the nearest angle value (numeric)', type=float, default=None)
    args = p.parse_args()

    path = Path(args.file)
    if not path.exists():
        print('File not found:', args.file)
        sys.exit(1)

    header_info, fields, data = parse_fes_file(path)
    x, y, reshaped = grid_from_data(data, header_info)

    # determine fields for reshaped last axis
    if reshaped.shape[2] == len(fields):
        fields_used = fields
    else:
        fields_used = []
        for i in range(reshaped.shape[2]):
            if i == 0:
                fields_used.append('sih')
            elif i == 1:
                fields_used.append('angle_phi')
            else:
                fields_used.append(f'col{i}')

    # choose z field
    if args.zfield:
        zfield_name = args.zfield
    else:
        prefer = ['file.free', 'fes', 'F', 'free_energy', 'free-energy', 'freeenergy']
        found = None
        for n in fields_used:
            if n in prefer:
                found = n
                break
        if found is None:
            for n in fields_used:
                if n not in ('sih', 'angle_phi', 'x', 'y'):
                    found = n
                    break
        if found is None:
            found = fields_used[2] if len(fields_used) >= 3 else fields_used[-1]
        zfield_name = found

    try:
        zidx = fields_used.index(zfield_name)
    except ValueError:
        zidx = 2 if len(fields_used) > 2 else -1

    Z = reshaped[:, :, zidx] * 0.010364  # Convert from kJ/mol to eV

    out_prefix = args.out if args.save else None
    # produce separate plots
    plot_3d_surface(x, y, Z, zlabel=zfield_name, show=not args.save, out_prefix=out_prefix)
    plot_contour(x, y, Z, zlabel=zfield_name, show=not args.save, out_prefix=out_prefix, annotate_minima=True)
    plot_1d_projections(x, y, Z, zlabel=zfield_name, show=not args.save, out_prefix=out_prefix, slice_angle=args.slice_angle)


if __name__ == '__main__':
    main()

