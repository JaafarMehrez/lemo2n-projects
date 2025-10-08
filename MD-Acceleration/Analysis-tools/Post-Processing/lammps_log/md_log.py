"""
Usage:
    python plot_lammps_quantity_vs_time.py log.lammps \
        [--time-unit {ps,fs,timestep}] [--timestep DT] [--units UNITS] \
        [--y-col COLNAME] [--smooth WINDOW] [--downsample N] \
        [--outfile FILE]
Examples:
    python plot_lammps_quantity_vs_time.py log.lammps --outfile energy_vs_time.png
    python plot_lammps_quantity_vs_time.py log.lammps --time-unit ps --y-col Volume
    python plot_lammps_quantity_vs_time.py log.lammps --timestep 0.5 --units real --y-col Press
Author: Jaafar Mehrez (jaafarmehrez@sjtu.edu.cn)
"""
import re
import argparse
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

def find_units_and_timestep(text):
    units_m = re.search(r'^\s*units\s+(\S+)', text, re.I | re.M)
    units = units_m.group(1).lower() if units_m else None
    ts_vals = [float(m) for m in re.findall(r'^\s*timestep\s+([0-9.eE+\-]+)', text, re.I | re.M)]
    return units, ts_vals  # ts_vals may have multiple entries

def read_lammps_log_numeric_blocks(filename):
    text = open(filename, 'r').read()
    lines = text.splitlines()
    header_indices = [i for i, line in enumerate(lines) if line.strip().startswith("Step")]
    if not header_indices:
        raise RuntimeError("No 'Step' header found in log file")
    collected = []
    header = None

    def is_number_token(tok):
        try:
            float(tok)
            return True
        except Exception:
            return False

    for hi in header_indices:
        hdr_tokens = lines[hi].split()
        if header is None:
            header = hdr_tokens
        j = hi + 1
        while j < len(lines):
            line = lines[j].strip()
            if not line:
                j += 1
                continue
            if line.startswith("Step"):
                break
            if not re.match(r'^[\s]*[+-]?\d', lines[j]):
                break
            tokens = lines[j].split()
            num_tokens = [t for t in tokens if is_number_token(t)]
            if len(num_tokens) >= len(header):
                collected.append(' '.join(num_tokens[:len(header)]))
            j += 1
            continue
        else:
            break
    if not collected:
        raise RuntimeError("No numeric thermo data found after any 'Step' header")
    df = pd.read_csv(StringIO('\n'.join(collected)), sep=r'\s+', names=header, engine='python')
    df = df.apply(pd.to_numeric, errors='coerce').dropna(how='all')
    return df, text

def convert_steps_to_time(df, logfile_text=None, timestep_override=None, units_override=None, target_time_unit='ps'):
    # detect units and timestep if not supplied
    units, ts_vals = find_units_and_timestep(logfile_text) if logfile_text is not None else (None, [])
    if units_override:
        units = units_override.lower()
    if timestep_override is not None:
        timestep = float(timestep_override)
    else:
        timestep = ts_vals[0] if ts_vals else None
    if timestep is None:
        raise RuntimeError("No timestep found in log and no --timestep provided. Provide timestep (e.g. 0.5) or add 'timestep <value>' to the input/log.")
    if len(ts_vals) > 1 and timestep_override is None:
        print("Warning: multiple 'timestep' entries found in the log. Using the first occurrence by default. "
              "If timestep changed during the run you need custom handling.")
    # mapping from LAMMPS units to factor that converts 1 time unit -> picoseconds
    mapping_to_ps = {
        'real': 1e-3,  # 1 fs = 1e-3 ps
        'metal': 1.0,  # 1 ps = 1 ps
        'si': 1e12,    # 1 s = 1e12 ps
        'cgs': 1e12,   # 1 s = 1e12 ps
        # 'lj' is reduced - cannot convert without model parameters
    }
    if units is None:
        print("Warning: 'units' not found in logfile. Assuming units correspond to given overrides (if provided).")
    elif units == 'lj':
        raise RuntimeError("units=lj (reduced units). Time is dimensionless; cannot auto-convert to physical units without sigma/epsilon.")
    if units not in mapping_to_ps and units_override is None:
        raise RuntimeError(f"Unknown or unsupported units='{units}'. Provide --units and --timestep explicitly.")
    # dt in ps:
    factor_to_ps = mapping_to_ps.get(units, None) if units_override is None else mapping_to_ps.get(units_override, None)
    if factor_to_ps is None:
        # if user provided units_override but it's not in mapping, error
        raise RuntimeError(f"Cannot convert units='{units}' to ps. Use --time-unit 'timestep' to plot in raw step*dt units instead.")
    dt_ps = float(timestep) * factor_to_ps
    df = df.copy()
    # raw time in timestep units (i.e., Step * dt as reported)
    df['Time_raw'] = df['Step'] * float(timestep)
    # provide requested unit
    if target_time_unit == 'ps':
        df['Time_ps'] = df['Time_raw'] * factor_to_ps
        time_col = 'Time_ps'
        time_label = 'Time (ps)'
    elif target_time_unit == 'fs':
        # 1 ps = 1000 fs
        df['Time_fs'] = df['Time_raw'] * factor_to_ps * 1000.0
        time_col = 'Time_fs'
        time_label = 'Time (fs)'
    elif target_time_unit == 'timestep':
        df['Time_tstep_units'] = df['Time_raw']
        time_col = 'Time_tstep_units'
        time_label = f'Time (timestep units; dt={timestep})'
    else:
        raise RuntimeError("Unknown target_time_unit. Use 'ps', 'fs', or 'timestep'.")
    return df, time_col, time_label

def auto_select_y_column(df, hint_list=None):
    if hint_list is None:
        hint_list = ['TotEng', 'PotEng', 'Etot', 'etotal', 'TotEnergy', 'PE', 'PotEnergy', 'TotEng']
    for c in hint_list:
        if c in df.columns:
            return c
    # fallback: choose 3rd column often (Step, Temp, Energy)
    if df.shape[1] >= 3:
        return df.columns[2]
    raise RuntimeError("Could not find a suitable column automatically. Use --y-col to specify the column name.")

def main():
    parser = argparse.ArgumentParser(description="Plot LAMMPS quantity vs time from log.lammps")
    parser.add_argument("logfile", help="LAMMPS log filename")
    parser.add_argument("--time-unit", choices=['ps', 'fs', 'timestep'], default='ps', help="plot time in ps, fs, or raw timestep units")
    parser.add_argument("--timestep", type=float, default=None, help="override timestep value (e.g., 0.5)")
    parser.add_argument("--units", type=str, default=None, help="override LAMMPS units (e.g., real, metal)")
    parser.add_argument("--y-col", type=str, default=None, help="name of column to plot (default: auto-detect energy column)")
    parser.add_argument("--smooth", type=int, default=0, help="apply simple moving average smoothing with window (in points)")
    parser.add_argument("--downsample", type=int, default=1, help="plot every N-th point (useful for large files)")
    parser.add_argument("--outfile", type=str, default=None, help="save figure to file instead of showing")
    args = parser.parse_args()

    df, logfile_text = read_lammps_log_numeric_blocks(args.logfile)
    try:
        df_conv, time_col, time_label = convert_steps_to_time(df, logfile_text=logfile_text,
                                                              timestep_override=args.timestep,
                                                              units_override=args.units,
                                                              target_time_unit=args.time_unit)
    except Exception as e:
        raise

    y_col = args.y_col if args.y_col else auto_select_y_column(df_conv)

    if y_col not in df_conv.columns:
        raise RuntimeError(f"Specified column '{y_col}' not found in the data. Available columns: {', '.join(df_conv.columns)}")

    # optionally smooth
    if args.smooth and args.smooth > 1:
        df_conv[y_col + '_sm'] = df_conv[y_col].rolling(window=args.smooth, center=True, min_periods=1).mean()
        plot_col = y_col + '_sm'
        label_extra = f"{y_col} (smoothed w={args.smooth})"
    else:
        plot_col = y_col
        label_extra = y_col

    # downsample for plotting if requested
    if args.downsample and args.downsample > 1:
        df_plot = df_conv.iloc[::args.downsample, :].reset_index(drop=True)
    else:
        df_plot = df_conv

    plt.figure(figsize=(6, 4))
    plt.plot(df_plot[time_col], df_plot[plot_col], label=label_extra, color='black')
    plt.xlabel(time_label, fontname="Georgia", fontsize=12)
    plt.ylabel(y_col, fontname="Georgia", fontsize=12)
    plt.title(f"{y_col} vs {time_label}", fontname="Georgia")
    plt.tick_params(axis='both', which='major', labelsize=10, labelfontfamily='Georgia')
    plt.grid(False)
    plt.legend(
        prop={'family': 'Georgia', 'size': 10},
        frameon=True,
        edgecolor='black',
        framealpha=1.0,
        loc='lower right'
    )
    plt.tight_layout()

    if args.outfile:
        plt.savefig(args.outfile, dpi=300)
        print(f"Saved figure to {args.outfile}")
    else:
        plt.show()

if __name__ == "__main__":
    main()