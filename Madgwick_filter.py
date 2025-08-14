# MADGWICK FILTER (independent version; saves to results_csv/)

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# === Load CSVs ===
imu_df = pd.read_csv('C:/Users/Hrigved/Desktop/ATTITUDE ESTIMATION/csv/imu1.csv')
mag_df = pd.read_csv('C:/Users/Hrigved/Desktop/ATTITUDE ESTIMATION/csv/mag1.csv')
# Ensure same length if inputs differ
N = min(len(imu_df), len(mag_df))
imu_df = imu_df.iloc[:N].reset_index(drop=True)
mag_df = mag_df.iloc[:N].reset_index(drop=True)

# === Time in seconds since start (handles ns or s) ===
t_raw = imu_df['%time'].to_numpy().astype(float)
if t_raw.max() > 1e8:  # likely nanoseconds
    time = (t_raw - t_raw.min()) / 1e9
else:  # already seconds
    time = t_raw - t_raw.min()

dt = np.diff(time)
dt = np.append(dt, dt[-1] if len(dt) else 0.0)  # keep same length

# === Madgwick Parameters ===
beta = 0.1  # Correction gain
q = np.array([1.0, 0.0, 0.0, 0.0])  # initial quaternion

# === Storage ===
rolls, pitches, yaws = [], [], []

# === Helper functions ===
def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v

def quat_mult(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# === Madgwick Filter Loop ===
for i in range(N):
    ax = imu_df['field.linear_acceleration.x'][i]
    ay = imu_df['field.linear_acceleration.y'][i]
    az = imu_df['field.linear_acceleration.z'][i]
    gx = imu_df['field.angular_velocity.x'][i]
    gy = imu_df['field.angular_velocity.y'][i]
    gz = imu_df['field.angular_velocity.z'][i]
    mx = mag_df['field.magnetic_field.x'][i]
    my = mag_df['field.magnetic_field.y'][i]
    mz = mag_df['field.magnetic_field.z'][i]
    dt_i = float(dt[i])

    acc = normalize(np.array([ax, ay, az], dtype=float))
    mag = normalize(np.array([mx, my, mz], dtype=float))

    # Step 1: Gyro quaternion derivative
    q_dot_gyro = 0.5 * quat_mult(q, np.array([0.0, gx, gy, gz]))

    # Step 2: Estimate gravity from quaternion
    g_est = np.array([
        2*(q[1]*q[3] - q[0]*q[2]),
        2*(q[0]*q[1] + q[2]*q[3]),
        q[0]**2 - q[1]**2 - q[2]**2 + q[3]**2
    ])

    # Step 3: Compute correction (accelerometer-only gradient descent)
    f = g_est - acc
    J = np.array([
        [-2*q[2],  2*q[3], -2*q[0],  2*q[1]],
        [ 2*q[1],  2*q[0],  2*q[3],  2*q[2]],
        [     0., -4*q[1], -4*q[2],      0.]
    ])
    grad_f = J.T @ f
    n = np.linalg.norm(grad_f)
    if n > 0:
        grad_f = grad_f / n

    # Step 4: Apply correction
    q_dot = q_dot_gyro - beta * grad_f

    # Step 5: Integrate and normalize
    q = q + q_dot * dt_i
    q = normalize(q)

    # Convert to roll, pitch, yaw
    rot = R.from_quat([q[1], q[2], q[3], q[0]])  # x, y, z, w
    eul = rot.as_euler('xyz', degrees=True)
    rolls.append(eul[0])
    pitches.append(eul[1])
    yaws.append(eul[2])

# === Plot All (Madgwick only) ===
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

axs[0].plot(time, rolls, label="Roll (Madgwick)", color='tab:blue')
axs[0].set_ylabel("Roll (°)")
axs[0].legend()
axs[0].grid()

axs[1].plot(time, pitches, label="Pitch (Madgwick)", color='tab:orange')
axs[1].set_ylabel("Pitch (°)")
axs[1].legend()
axs[1].grid()

axs[2].plot(time, yaws, label="Yaw (Madgwick)", color='tab:green')
axs[2].set_ylabel("Yaw (°)")
axs[2].set_xlabel("Time (s)")
axs[2].legend()
axs[2].grid()

plt.suptitle("Madgwick Filter Output (Roll, Pitch, Yaw)")
plt.tight_layout()
plt.show()

# === Save Madgwick Euler outputs to results_csv/ ===
os.makedirs("result_csv", exist_ok=True)
out_madg = pd.DataFrame({
    'timestamp': time,
    'roll': rolls,
    'pitch': pitches,
    'yaw': yaws
})
out_path = 'result_csv/madgwick_euler.csv'
out_madg.to_csv(out_path, index=False)
print(f"[OK] Saved Madgwick Euler → {out_path}")
