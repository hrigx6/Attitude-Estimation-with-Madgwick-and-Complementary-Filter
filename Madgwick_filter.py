# ========= Madgwick (IMU + yaw-only MAG) with calibrated MAG + overlay =========
import numpy as np, json
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# --- helper: get calibrated mag if available, else raw ---
def get_calibrated_mag_arrays(mag_df, cal_json_path="mag_cal.json"):
    # Prefer precomputed columns if your calibration CSV was merged back
    if all(c in mag_df.columns for c in ["mag_x_cal","mag_y_cal","mag_z_cal"]):
        return (mag_df["mag_x_cal"].to_numpy(float),
                mag_df["mag_y_cal"].to_numpy(float),
                mag_df["mag_z_cal"].to_numpy(float))
    # Else try mag_cal.json
    def raw(col):
        if f"magnetic_field_{col}" in mag_df.columns:
            return mag_df[f"magnetic_field_{col}"].to_numpy(float)
        return mag_df[f"field.magnetic_field.{col}"].to_numpy(float)
    mx_raw, my_raw, mz_raw = raw("x"), raw("y"), raw("z")
    try:
        with open(cal_json_path, "r") as f:
            p = json.load(f)
        b = np.asarray(p["offset"], float)
        A = np.asarray(p["matrix"], float)
        Mcal = (A @ (np.column_stack([mx_raw,my_raw,mz_raw]) - b).T).T
        return Mcal[:,0], Mcal[:,1], Mcal[:,2]
    except Exception:
        return mx_raw, my_raw, mz_raw

# --- Madgwick (IMU + yaw-only MAG) ---
def run_madgwick6_with_yaw_mag(imu_df, mag_df, beta=0.07, fc_yaw=2.0, gate_frac=0.35):
    def sec(df):
        if 'timestamp' in df.columns:
            t = df['timestamp'].to_numpy(float); return t - np.nanmin(t)
        t = df['%time'].to_numpy(float); s = 1e9 if np.nanmax(t) > 1e11 else 1.0
        return (t - np.nanmin(t))/s
    def col(df, base):
        def pick(x, alt): return df[x].to_numpy(float) if x in df.columns else df[alt].to_numpy(float)
        return (pick(f"{base}_x", f"field.{base}.x"),
                pick(f"{base}_y", f"field.{base}.y"),
                pick(f"{base}_z", f"field.{base}.z"))
    def norm(v):
        n = np.linalg.norm(v); return v/n if n>0 else v
    def qmul(q1,q2):
        w1,x1,y1,z1=q1; w2,x2,y2,z2=q2
        return np.array([w1*w2-x1*x2-y1*y2-z1*z2,
                         w1*x2+x1*w2+y1*z2-z1*y2,
                         w1*y2-x1*z2+y1*w2+z1*x2,
                         w1*z2+x1*y2-y1*x2+z1*w2])
    def wrap_pi(a): return (a+np.pi)%(2*np.pi)-np.pi

    t = sec(imu_df)
    dt = np.diff(t); dt = np.append(dt, dt[-1] if len(dt) else 0.0)

    ax,ay,az = col(imu_df, "linear_acceleration")
    gx,gy,gz = col(imu_df, "angular_velocity")
    mx,my,mz  = get_calibrated_mag_arrays(mag_df)

    N = min(len(t), len(ax), len(mx))
    t=t[:N]; dt=dt[:N]
    ax,ay,az = ax[:N],ay[:N],az[:N]
    gx,gy,gz = gx[:N],gy[:N],gz[:N]
    mx,my,mz = mx[:N],my[:N],mz[:N]

    # deg/s → rad/s if likely
    if np.nanmax(np.abs([gx,gy,gz])) > 50:
        gx,gy,gz = np.radians(gx), np.radians(gy), np.radians(gz)

    mnorm = np.sqrt(mx*mx + my*my + mz*mz)
    mref  = np.nanmedian(mnorm) if np.isfinite(mnorm).any() else 1.0
    mlo, mhi = (1-gate_frac)*mref, (1+gate_frac)*mref

    q = np.array([1.0,0.0,0.0,0.0], float)
    roll = np.empty(N); pitch = np.empty(N); yaw = np.empty(N)

    for i in range(N):
        dt_i = float(dt[i])

        # IMU step (accel-only gradient)
        acc = norm(np.array([ax[i],ay[i],az[i]]))
        q_dot_gyro = 0.5 * qmul(q, np.array([0.0,gx[i],gy[i],gz[i]]))
        w,x,y,z = q
        g_est = np.array([2*(x*z - w*y), 2*(w*x + y*z), w*w - x*x - y*y + z*z])
        f = g_est - acc
        J = np.array([[-2*y, 2*z,-2*w, 2*x],
                      [ 2*x, 2*w, 2*z, 2*y],
                      [ 0.0,-4*x,-4*y, 0.0]])
        grad = J.T @ f
        n = np.linalg.norm(grad)
        if n>0: grad /= n
        q_dot = q_dot_gyro - beta*grad
        q = q + q_dot * dt_i; q /= np.linalg.norm(q)

        # yaw-only mag correction (tilt-compensated), gated & low-pass blended
        if mlo <= mnorm[i] <= mhi and np.all(np.isfinite([mx[i], my[i], mz[i]])):
            w,x,y,z = q
            R00 = 1-2*(y*y+z*z); R01 = 2*(x*y - w*z); R02 = 2*(x*z + w*y)
            R10 = 2*(x*y + w*z); R11 = 1-2*(x*x+z*z); R12 = 2*(y*z - w*x)
            mx_e = R00*mx[i] + R01*my[i] + R02*mz[i]
            my_e = R10*mx[i] + R11*my[i] + R12*mz[i]
            yaw_mag  = np.arctan2(-my_e, mx_e)
            yaw_curr = R.from_quat([x,y,z,w]).as_euler('xyz', degrees=False)[2]
            dyaw = wrap_pi(yaw_mag - yaw_curr)

            tau = 1.0/(2*np.pi*fc_yaw)
            alpha_y = tau/(tau + dt_i)
            d = (1 - alpha_y) * dyaw
            dq = np.array([np.cos(d/2), 0.0, 0.0, np.sin(d/2)])
            q  = qmul(dq, q); q /= np.linalg.norm(q)

        e = R.from_quat([q[1],q[2],q[3],q[0]]).as_euler('xyz', degrees=True)
        roll[i], pitch[i], yaw[i] = e

    yaw = np.degrees(np.unwrap(np.radians(yaw)))
    return t, roll, pitch, yaw

# ---------- run & overlay ----------
t_m, roll_m, pitch_m, yaw_m = run_madgwick6_with_yaw_mag(imu_df, mag_df,
                                                         beta=0.07, fc_yaw=2.0, gate_frac=0.35)

# Complementary arrays are *already computed* earlier in the notebook:
#   roll_list, pitch_list, yaw_list_unwrapped, imu_time
# Interpolate complementary to Madgwick time for a clean overlay.
t_comp = imu_time.values[1:1+len(roll_list)]  # matches how you built complementary outputs

def interp_to_ref(t_src, y_src, t_ref):
    y_out = np.interp(np.clip(t_ref, t_src[0], t_src[-1]), t_src, y_src)
    # mask outside overlap to NaN (optional; comment out if you prefer extrapolated lines)
    mask = (t_ref >= t_src[0]) & (t_ref <= t_src[-1])
    y_out[~mask] = np.nan
    return y_out

roll_c_i = interp_to_ref(t_comp, np.asarray(roll_list), t_m)
pitch_c_i= interp_to_ref(t_comp, np.asarray(pitch_list), t_m)
yaw_c_i  = interp_to_ref(t_comp, np.asarray(yaw_list_unwrapped), t_m)

# Plot: Madgwick only + overlay with complementary
fig, axs = plt.subplots(3,1, figsize=(12,10), sharex=True)

axs[0].plot(t_m, roll_m, label='Roll — Madgwick', lw=1.2)
axs[0].plot(t_m, roll_c_i, label='Roll — Complementary', lw=1.0, alpha=0.85)
axs[0].set_ylabel('Roll (°)'); axs[0].legend(); axs[0].grid(True, alpha=0.3)

axs[1].plot(t_m, pitch_m, label='Pitch — Madgwick', lw=1.2)
axs[1].plot(t_m, pitch_c_i, label='Pitch — Complementary', lw=1.0, alpha=0.85)
axs[1].set_ylabel('Pitch (°)'); axs[1].legend(); axs[1].grid(True, alpha=0.3)

axs[2].plot(t_m, yaw_m, label='Yaw — Madgwick', lw=1.2)
axs[2].plot(t_m, yaw_c_i, label='Yaw — Complementary', lw=1.0, alpha=0.85)
axs[2].set_ylabel('Yaw (°)'); axs[2].set_xlabel('Time (s)')
axs[2].legend(); axs[2].grid(True, alpha=0.3)

plt.tight_layout(); plt.show()
