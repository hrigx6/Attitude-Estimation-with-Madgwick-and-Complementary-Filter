#COMPARISION COMP VS MAD

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import atan2, asin
from pathlib import Path
from dataclasses import dataclass

# --------- helpers ---------
def wrap_deg(delta_deg):
    """Minimal signed angle difference in degrees [-180,180)."""
    d = (delta_deg + 180.0) % 360.0 - 180.0
    return d

def euler_deg_to_quat(roll, pitch, yaw):
    """XYZ convention, degrees -> quaternion [w,x,y,z]."""
    r = np.radians(roll); p = np.radians(pitch); y = np.radians(yaw)
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    w = cr*cp*cy + sr*cp*sy
    x = sr*cp*cy - cr*sp*sy
    yq = cr*sp*cy + sr*sp*sy
    z = cr*cp*sy - sr*cp*cy
    q = np.stack([w,x,yq,z], axis=-1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    return q

def quat_distance_deg(q_ref, q_est):
    """
    Geodesic quaternion distance in degrees.
    q_ref, q_est: arrays [N,4] with [w,x,y,z]
    """
    # handle double cover: take absolute dot
    dots = np.sum(q_ref*q_est, axis=1)
    dots = np.clip(np.abs(dots), -1.0, 1.0)
    return np.degrees(2*np.arccos(dots))

def interp_to(t_src, y_src, t_tgt):
    """1D interpolation per column to target time base."""
    y_src = np.asarray(y_src)
    if y_src.ndim == 1:  # shape [N]
        return np.interp(t_tgt, t_src, y_src)
    # shape [N, C]
    out = np.empty((len(t_tgt), y_src.shape[1]))
    for c in range(y_src.shape[1]):
        out[:, c] = np.interp(t_tgt, t_src, y_src[:, c])
    return out

@dataclass
class Series:
    t: np.ndarray
    roll: np.ndarray  # deg
    pitch: np.ndarray # deg
    yaw: np.ndarray   # deg

# --------- load data ---------
# Expected CSVs (rename paths if needed):
PX4_CSV = "C:/Users/Hrigved/Desktop/ATTITUDE ESTIMATION/result_csv/px4_attitude.csv"          # columns: timestamp,qw,qx,qy,qz
COMP_CSV = "C:/Users/Hrigved/Desktop/ATTITUDE ESTIMATION/result_csv/complementary_euler.csv"   # columns: timestamp,roll,pitch,yaw (deg)
MADG_CSV = "C:/Users/Hrigved/Desktop/ATTITUDE ESTIMATION/result_csv/madgwick_euler.csv"        # columns: timestamp,roll,pitch,yaw (deg)

def load_px4(csv_path):
    df = pd.read_csv(csv_path)
    t = (df['timestamp'] - df['timestamp'].min()).to_numpy().astype(float) / 1e9 if df['timestamp'].max() > 1e8 else df['timestamp'].to_numpy().astype(float)
    q = df[['qw','qx','qy','qz']].to_numpy().astype(float)
    # Euler (deg) from quaternion (XYZ convention)
    # Implement here to avoid SciPy dependency
    w,x,y,z = q.T
    # roll
    sinr = 2*(w*x + y*z)
    cosr = 1 - 2*(x*x + y*y)
    roll = np.degrees(np.arctan2(sinr, cosr))
    # pitch
    sinp = 2*(w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))
    # yaw
    siny = 2*(w*z + x*y)
    cosy = 1 - 2*(y*y + z*z)
    yaw = np.degrees(np.arctan2(siny, cosy))
    return t, q, Series(t, roll, pitch, yaw)

def load_euler(csv_path):
    df = pd.read_csv(csv_path)
    t = (df['timestamp'] - df['timestamp'].min()).to_numpy().astype(float) / 1e9 if df['timestamp'].max() > 1e8 else df['timestamp'].to_numpy().astype(float)
    return Series(t,
                  df['roll'].to_numpy().astype(float),
                  df['pitch'].to_numpy().astype(float),
                  df['yaw'].to_numpy().astype(float))

# --------- metrics ---------
def per_axis_metrics(ref: Series, est: Series):
    # Interpolate estimates to reference time (PX4)
    roll_i  = interp_to(est.t, est.roll,  ref.t)
    pitch_i = interp_to(est.t, est.pitch, ref.t)
    yaw_i   = interp_to(est.t, est.yaw,   ref.t)

    err_roll  = roll_i  - ref.roll
    err_pitch = pitch_i - ref.pitch
    err_yaw   = wrap_deg(yaw_i - ref.yaw)

    mae = np.array([np.mean(np.abs(err_roll)),
                    np.mean(np.abs(err_pitch)),
                    np.mean(np.abs(err_yaw))])
    rmse = np.sqrt(np.array([np.mean(err_roll**2),
                             np.mean(err_pitch**2),
                             np.mean(err_yaw**2)]))
    return mae, rmse, (err_roll, err_pitch, err_yaw)

def overall_quat_error_deg(ref_q, est_euler_on_ref_time: Series):
    # Build quats from Euler estimates (on ref timebase already)
    q_est = euler_deg_to_quat(est_euler_on_ref_time.roll,
                              est_euler_on_ref_time.pitch,
                              est_euler_on_ref_time.yaw)
    return quat_distance_deg(ref_q, q_est)

# --------- main ---------
def main():
    out_dir = Path("plots"); out_dir.mkdir(parents=True, exist_ok=True)

    # Load
    t_px4, q_px4, px4 = load_px4(PX4_CSV)
    comp = load_euler(COMP_CSV)
    madg = load_euler(MADG_CSV)

    # Interpolate Euler to PX4 time
    comp_on_ref = Series(
        t_px4,
        interp_to(comp.t, comp.roll, t_px4),
        interp_to(comp.t, comp.pitch, t_px4),
        interp_to(comp.t, comp.yaw, t_px4),
    )
    madg_on_ref = Series(
        t_px4,
        interp_to(madg.t, madg.roll, t_px4),
        interp_to(madg.t, madg.pitch, t_px4),
        interp_to(madg.t, madg.yaw, t_px4),
    )

    # Metrics
    comp_mae, comp_rmse, comp_errs = per_axis_metrics(px4, comp_on_ref)
    madg_mae, madg_rmse, madg_errs = per_axis_metrics(px4, madg_on_ref)

    comp_qerr = overall_quat_error_deg(q_px4, comp_on_ref)
    madg_qerr = overall_quat_error_deg(q_px4, madg_on_ref)

    # Save metrics CSV
    metrics = pd.DataFrame({
        'method': ['Complementary','Complementary','Complementary','Madgwick','Madgwick','Madgwick'],
        'metric': ['MAE','RMSE','QuatMeanDeg','MAE','RMSE','QuatMeanDeg'],
        'roll_deg': [comp_mae[0], comp_rmse[0], np.nan, madg_mae[0], madg_rmse[0], np.nan],
        'pitch_deg':[comp_mae[1], comp_rmse[1], np.nan, madg_mae[1], madg_rmse[1], np.nan],
        'yaw_deg':  [comp_mae[2], comp_rmse[2], np.nan, madg_mae[2], madg_rmse[2], np.nan],
        'overall_deg':[np.nan, np.nan, float(np.mean(comp_qerr)), np.nan, np.nan, float(np.mean(madg_qerr))]
    })
    metrics.to_csv(out_dir / "metrics.csv", index=False)

    # Plots: errors vs time
    def plot_err(time, errs, title, save):
        labels = ['Roll error (deg)','Pitch error (deg)','Yaw error (deg)']
        plt.figure(figsize=(12,6))
        for i,e in enumerate(errs):
            plt.plot(time, e, label=labels[i])
        plt.axhline(0, linewidth=0.8)
        plt.xlabel('Time (s)'); plt.ylabel('Error (deg)'); plt.title(title)
        plt.legend(); plt.grid(True, alpha=0.3)
        plt.tight_layout(); plt.savefig(save); plt.close()

    plot_err(t_px4, comp_errs, "Complementary vs PX4 — per-axis error", out_dir/"comp_errors.png")
    plot_err(t_px4, madg_errs, "Madgwick vs PX4 — per-axis error", out_dir/"madg_errors.png")

    # Print summary
    def fmt(a): return ', '.join(f'{x:.3f}' for x in a)
    print("Complementary — MAE [roll,pitch,yaw] deg:", fmt(comp_mae))
    print("Complementary — RMSE [roll,pitch,yaw] deg:", fmt(comp_rmse))
    print("Complementary — Mean quaternion angular error deg:", np.mean(comp_qerr).round(3))
    print()
    print("Madgwick — MAE [roll,pitch,yaw] deg:", fmt(madg_mae))
    print("Madgwick — RMSE [roll,pitch,yaw] deg:", fmt(madg_rmse))
    print("Madgwick — Mean quaternion angular error deg:", np.mean(madg_qerr).round(3))
    print("\nSaved:", str(out_dir / "metrics.csv"))
    print("Plots:", str(out_dir / "comp_errors.png"), ",", str(out_dir / "madg_errors.png"))

if __name__ == "__main__":
    main()
