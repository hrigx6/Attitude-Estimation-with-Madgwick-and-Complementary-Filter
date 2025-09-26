import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, sqrt, degrees
from scipy.spatial.transform import Rotation as R


# Funtion to use calibrated mag data if available else proceed with raw data
import json

def get_calibrated_mag_arrays(mag_df, cal_json_path="mag_cal.json", return_source=False):
    """
    Returns (mx, my, mz) as float arrays.
    Priority:
      1) If DataFrame has ['mag_x_cal','mag_y_cal','mag_z_cal'] -> use those.
      2) Else, if mag_cal.json exists -> apply A @ (raw - b) and return.
      3) Else -> return raw ['magnetic_field_*' or 'field.magnetic_field.*'].
    """
    
    if all(c in mag_df.columns for c in ["mag_x_cal","mag_y_cal","mag_z_cal"]):
        mx = mag_df["mag_x_cal"].to_numpy(float)
        my = mag_df["mag_y_cal"].to_numpy(float)
        mz = mag_df["mag_z_cal"].to_numpy(float)
        return (mx, my, mz, "df:cal_cols") if return_source else (mx, my, mz)

    # raw getters
    def get_raw(col_short):
        if f"magnetic_field_{col_short}" in mag_df.columns:
            return mag_df[f"magnetic_field_{col_short}"].to_numpy(float)
        return mag_df[f"field.magnetic_field.{col_short}"].to_numpy(float)

    mx_raw = get_raw("x"); my_raw = get_raw("y"); mz_raw = get_raw("z")

    # 2) try mag_cal.json
    try:
        with open(cal_json_path, "r") as f:
            p = json.load(f)
        b = np.asarray(p["offset"], float)
        A = np.asarray(p["matrix"], float)
        Mraw = np.column_stack([mx_raw, my_raw, mz_raw])
        Mcal = (A @ (Mraw - b).T).T
        mx, my, mz = Mcal[:,0], Mcal[:,1], Mcal[:,2]
        return (mx, my, mz, "json:mag_cal") if return_source else (mx, my, mz)
    except Exception:
        # 3) fallback: raw
        return (mx_raw, my_raw, mz_raw, "df:raw") if return_source else (mx_raw, my_raw, mz_raw)



 
### User controls [change the params to adjust the filter]

# Choose a time window to visualize (seconds from start). Set both to None to plot the full run.
SEGMENT_START_SEC = 30   # e.g., 5.0
SEGMENT_END_SEC   = 60   # e.g., 15.0

# Decimate plotted samples to reduce density (1 = no decimation, 5 = every 5th point, etc.)
PLOT_EVERY_N = 1

# Complementary filter cutoffs (Hz). Higher cutoff => more accel/mag trust (less gyro dominance)
FC_ROLL_PITCH = 10.0
FC_YAW        = 2.0

# Accel gating: reduce accel trust when |a| is far from g (during linear accel)
USE_ACCEL_GATING = True
G = 9.80665
ACCEL_GATING_TOL = 2.0  # m/s^2 tolerance around g


### Load data
imu_df = pd.read_csv('/content/main_bag_imu.csv').sort_values('timestamp').reset_index(drop=True)
mag_df = pd.read_csv('/content/main_bag_mag.csv').sort_values('timestamp').reset_index(drop=True)
ekf_df = pd.read_csv('/content/main_bag_ekf.csv').sort_values('timestamp').reset_index(drop=True)

imu_time = imu_df['timestamp'] - imu_df['timestamp'].min()
mag_time = mag_df['timestamp'] - mag_df['timestamp'].min()

# Match IMU & MAG lengths
length = min(len(imu_df), len(mag_df)) - 1
dt = np.diff(imu_time.values)

# Get magnetometer arrays from the mag_calibrated if available function
mx_arr, my_arr, mz_arr, mag_src = get_calibrated_mag_arrays(mag_df, cal_json_path="mag_cal.json", return_source=True)
print("Mag source:", mag_src)


# =========================
# Complementary filter
# =========================
def alpha_from_fc(fc, dt_s):
    tau = 1.0 / (2.0 * np.pi * fc)           # time constant
    return tau / (tau + dt_s)                # α closer to 1 => more gyro, closer to 0 => more accel/mag

roll = pitch = yaw = 0.0

roll_list, pitch_list, yaw_list = [], [], []
roll_accel_list, pitch_accel_list, yaw_mag_list = [], [], []
roll_gyro_list, pitch_gyro_list, yaw_gyro_list = [], [], []

for i in range(1, length):
    # IMU
    ax = imu_df['linear_acceleration_x'].iloc[i]
    ay = imu_df['linear_acceleration_y'].iloc[i]
    az = imu_df['linear_acceleration_z'].iloc[i]
    gx = imu_df['angular_velocity_x'].iloc[i]
    gy = imu_df['angular_velocity_y'].iloc[i]
    gz = imu_df['angular_velocity_z'].iloc[i]

    
    # MAG 
    mx = mx_arr[i]; my = my_arr[i]; mz = mz_arr[i]
 

    dt_i = float(dt[i - 1]) if i - 1 < len(dt) else 0.0

    # accel -> roll/pitch
    pitch_a = atan2(-ax, sqrt(ay**2 + az**2))
    roll_a  = atan2(ay, az)

    # gyro integrate
    pitch_g = pitch + gy * dt_i
    roll_g  = roll  + gx * dt_i
    yaw_g   = yaw   + gz * dt_i

    # tilt-compensated heading from mag
    Xh = mx * np.cos(pitch) + mz * np.sin(pitch)
    Yh = mx * np.sin(roll) * np.sin(pitch) + my * np.cos(roll) - mz * np.sin(roll) * np.cos(pitch)
    yaw_m = np.arctan2(-Yh, Xh)

    # dynamic α from cutoffs
    alpha_rp = alpha_from_fc(FC_ROLL_PITCH, dt_i)
    alpha_y  = alpha_from_fc(FC_YAW,        dt_i)

    # accel gating
    if USE_ACCEL_GATING:
        acc_norm = sqrt(ax*ax + ay*ay + az*az)
        if abs(acc_norm - G) > ACCEL_GATING_TOL:
            scale = min(1.0, (abs(acc_norm - G) - ACCEL_GATING_TOL) / (2.0 * ACCEL_GATING_TOL))
            alpha_rp = alpha_rp + (1.0 - alpha_rp) * scale  # nudge toward gyro when accel unreliable

    # complementary update
    pitch = alpha_rp * pitch_g + (1.0 - alpha_rp) * pitch_a
    roll  = alpha_rp * roll_g  + (1.0 - alpha_rp) * roll_a
    yaw   = alpha_y  * yaw_g   + (1.0 - alpha_y)  * yaw_m

    roll_accel_list.append(degrees(roll_a))
    pitch_accel_list.append(degrees(pitch_a))
    yaw_mag_list.append(degrees(yaw_m))
    roll_gyro_list.append(degrees(roll_g))
    pitch_gyro_list.append(degrees(pitch_g))
    yaw_gyro_list.append(degrees(yaw_g))
    roll_list.append(degrees(roll))
    pitch_list.append(degrees(pitch))
    yaw_list.append(degrees(yaw))

# Unwrap yaw 
yaw_list_unwrapped = np.degrees(np.unwrap(np.radians(yaw_list)))
yaw_mag_unwrapped  = np.degrees(np.unwrap(np.radians(yaw_mag_list)))
yaw_gyro_unwrapped = np.degrees(np.unwrap(np.radians(yaw_gyro_list)))

# Time vector aligned with the complementary arrays
t_imu = imu_time.values[1:length]


# Choose a segment (or full run) and decimate from above set params (USER CONTROLS section)

def select_segment(t, *arrays, start=SEGMENT_START_SEC, end=SEGMENT_END_SEC, every=PLOT_EVERY_N):
    if start is None or end is None:
        mask = np.ones_like(t, dtype=bool)
    else:
        mask = (t >= float(start)) & (t <= float(end))

    idx = np.nonzero(mask)[0]
    if len(idx) == 0:
        # fall back to full if segment empty
        idx = np.arange(len(t))

    # decimate
    idx = idx[::max(1, int(every))]

    t_sel = t[idx]
    arrays_sel = [np.asarray(a)[idx] for a in arrays]
    return (t_sel, *arrays_sel)

(t_plot,
 roll_filt_p, roll_acc_p, roll_gyro_p,
 pitch_filt_p, pitch_acc_p, pitch_gyro_p,
 yaw_filt_p, yaw_mag_p, yaw_gyro_p) = select_segment(
    t_imu,
    roll_list, roll_accel_list, roll_gyro_list,
    pitch_list, pitch_accel_list, pitch_gyro_list,
    yaw_list_unwrapped, yaw_mag_unwrapped, yaw_gyro_unwrapped
)

# =========================
# Plotting 
# =========================
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Roll
axs[0].plot(t_plot, roll_filt_p, label='Roll (Complementary)', linewidth=1)
axs[0].plot(t_plot, roll_acc_p,  label='Roll (Accel-derived)', linewidth=1)
axs[0].plot(t_plot, roll_gyro_p, label='Roll (Gyro integ.)',   linewidth=1)
axs[0].set_ylabel('Roll (°)')
axs[0].legend(); axs[0].grid(True)

# Pitch
axs[1].plot(t_plot, pitch_filt_p, label='Pitch (Complementary)', linewidth=1)
axs[1].plot(t_plot, pitch_acc_p,  label='Pitch (Accel-derived)', linewidth=1)
axs[1].plot(t_plot, pitch_gyro_p, label='Pitch (Gyro integ.)',   linewidth=1)
axs[1].set_ylabel('Pitch (°)')
axs[1].legend(); axs[1].grid(True)

# Yaw
axs[2].plot(t_plot, yaw_filt_p, label='Yaw (Complementary)', linewidth=1)
axs[2].plot(t_plot, yaw_mag_p,  label='Yaw (Mag-derived)',   linewidth=1)
axs[2].plot(t_plot, yaw_gyro_p, label='Yaw (Gyro integ.)',   linewidth=1)
axs[2].set_ylabel('Yaw (°)')
axs[2].set_xlabel('Time (s)')
axs[2].legend(); axs[2].grid(True)

plt.tight_layout()
plt.show()


# Save outputs
out_comp = pd.DataFrame({
    'timestamp': imu_time.values[1:length],
    'roll_deg':  roll_list,
    'pitch_deg': pitch_list,
    'yaw_deg':   yaw_list_unwrapped
})
out_comp.to_csv('complementary_euler.csv', index=False)

# EKF quaternions (kept for downstream use; not plotted)
ekf_out = ekf_df[['timestamp','orientation_w','orientation_x','orientation_y','orientation_z']].copy()
ekf_out['timestamp'] = ekf_out['timestamp'] - ekf_out['timestamp'].min()
ekf_out = ekf_out.rename(columns={'orientation_w':'qw','orientation_x':'qx','orientation_y':'qy','orientation_z':'qz'})
ekf_out.to_csv('px4_attitude.csv', index=False)
