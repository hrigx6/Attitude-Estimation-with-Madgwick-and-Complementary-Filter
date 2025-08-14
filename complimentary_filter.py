#COMPLIMENTARY FILTER for PX4 ardupilot pixhawk

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import atan2, sqrt, degrees, radians, sin, cos
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt

# Load CSVs 
imu_df = pd.read_csv('C:/Users/Hrigved/Desktop/ATTITUDE ESTIMATION/csv/imu1.csv')
mag_df = pd.read_csv('C:/Users/Hrigved/Desktop/ATTITUDE ESTIMATION/csv/mag1.csv')
ekf_df = pd.read_csv('C:/Users/Hrigved/Desktop/ATTITUDE ESTIMATION/csv/ekf_data1.csv')

# Time Handling 
imu_time = (imu_df['%time'] - imu_df['%time'].min()) / 1e9
dt = np.diff(imu_time)

# Initialize 
alpha = 0.90
roll, pitch, yaw = 0, 0, 0

roll_list, pitch_list, yaw_list = [], [], []
roll_accel_list, pitch_accel_list, yaw_mag_list = [], [], []
roll_gyro_list, pitch_gyro_list, yaw_gyro_list = [], [], []


def butter_lowpass(cutoff, fs, order=3):
    return butter(order, cutoff / (0.5 * fs), btype='low', analog=False)

def butter_highpass(cutoff, fs, order=3):
    return butter(order, cutoff / (0.5 * fs), btype='high', analog=False)

# Complementary Filter 
length = min(len(imu_df), len(mag_df)) - 1
for i in range(1, length):
    ax = imu_df['field.linear_acceleration.x'][i]
    ay = imu_df['field.linear_acceleration.y'][i]
    az = imu_df['field.linear_acceleration.z'][i]
    gx = imu_df['field.angular_velocity.x'][i]
    gy = imu_df['field.angular_velocity.y'][i]
    gz = imu_df['field.angular_velocity.z'][i]
    mx = mag_df['field.magnetic_field.x'][i]
    my = mag_df['field.magnetic_field.y'][i]
    mz = mag_df['field.magnetic_field.z'][i]

    dt_i = dt[i - 1]

    pitch_a = atan2(-ax, sqrt(ay**2 + az**2))
    roll_a = atan2(ay, az)

    pitch_g = pitch + gy * dt_i
    roll_g = roll + gx * dt_i
    yaw_g = yaw + gz * dt_i

    pitch = alpha * pitch_g + (1 - alpha) * pitch_a
    roll = alpha * roll_g + (1 - alpha) * roll_a

    Xh = mx * cos(pitch) + mz * sin(pitch)
    Yh = mx * sin(roll) * sin(pitch) + my * cos(roll) - mz * sin(roll) * cos(pitch)
    yaw_m = atan2(-Yh, Xh)
    yaw = alpha * yaw_g + (1 - alpha) * yaw_m

    roll_accel_list.append(degrees(roll_a))
    pitch_accel_list.append(degrees(pitch_a))
    yaw_mag_list.append(degrees(yaw_m))
    roll_gyro_list.append(degrees(roll_g))
    pitch_gyro_list.append(degrees(pitch_g))
    yaw_gyro_list.append(degrees(yaw_g))
    roll_list.append(degrees(roll))
    pitch_list.append(degrees(pitch))
    yaw_list.append(degrees(yaw))

# EKF Orientation to Euler 
ekf_time = (ekf_df['%time'] - ekf_df['%time'].min()) / 1e9
ekf_roll, ekf_pitch, ekf_yaw = [], [], []

for i in range(len(ekf_df)):
    qx = ekf_df['field.orientation.x'][i]
    qy = ekf_df['field.orientation.y'][i]
    qz = ekf_df['field.orientation.z'][i]
    qw = ekf_df['field.orientation.w'][i]
    r = R.from_quat([qx, qy, qz, qw])
    (roll, pitch, yaw) = r.as_euler('xyz', degrees=True)

    ekf_roll.append(roll)
    ekf_pitch.append(pitch)
    ekf_yaw.append(yaw)

# Unwrap Yaw 
yaw_list_unwrapped = np.degrees(np.unwrap(np.radians(yaw_list)))
yaw_mag_unwrapped = np.degrees(np.unwrap(np.radians(yaw_mag_list)))
yaw_gyro_unwrapped = np.degrees(np.unwrap(np.radians(yaw_gyro_list)))
ekf_yaw_unwrapped = np.degrees(np.unwrap(np.radians(ekf_yaw)))



# Plotting
def plot_all_angles(time, roll_data, pitch_data, yaw_data, ekf_data):
    fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Roll
    axs[0].plot(time, roll_data['filt'], label='Roll (Complementary)', color='black', linewidth=1)
    axs[0].plot(time, roll_data['raw1'], label='Roll (Accel/Mag)', linestyle='-', linewidth=1)
    axs[0].plot(time, roll_data['raw2'], label='Roll (Gyro)', linestyle='-', linewidth=1)
    axs[0].plot(time[:len(ekf_data['roll'])], ekf_data['roll'], label='Roll (EKF)', linestyle='-', linewidth=1)
    axs[0].set_ylabel('Roll (°)')
    axs[0].legend()
    axs[0].grid(True)

    # Pitch
    axs[1].plot(time, pitch_data['filt'], label='Pitch (Complementary)', color='black', linewidth=1)
    axs[1].plot(time, pitch_data['raw1'], label='Pitch (Accel/Mag)', linestyle='-', linewidth=1)
    axs[1].plot(time, pitch_data['raw2'], label='Pitch (Gyro)', linestyle='-', linewidth=1)
    axs[1].plot(time[:len(ekf_data['pitch'])], ekf_data['pitch'], label='Pitch (EKF)', linestyle='-', linewidth=1)
    axs[1].set_ylabel('Pitch (°)')
    axs[1].legend()
    axs[1].grid(True)

    # Yaw
    axs[2].plot(time, yaw_data['filt'], label='Yaw (Complementary)', color='black', linewidth=1)
    axs[2].plot(time, yaw_data['raw1'], label='Yaw (Accel/Mag)', linestyle='-', linewidth=1)
    axs[2].plot(time, yaw_data['raw2'], label='Yaw (Gyro)', linestyle='-', linewidth=1)
    axs[2].plot(time[:len(ekf_data['yaw'])], ekf_data['yaw'], label='Yaw (EKF)', linestyle='-', linewidth=1)
    axs[2].set_ylabel('Yaw (°)')
    axs[2].set_xlabel('Time (s)')
    axs[2].legend()
    axs[2].grid(True)

    plt.tight_layout()
    plt.show()


plot_all_angles(
    imu_time[1:length],
    roll_data={
        'filt': roll_list,
        'raw1': roll_accel_list,
        'raw2': roll_gyro_list
    },
    pitch_data={
        'filt': pitch_list,
        'raw1': pitch_accel_list,
        'raw2': pitch_gyro_list
    },
    yaw_data={
        'filt': yaw_list_unwrapped,
        'raw1': yaw_mag_unwrapped,
        'raw2': yaw_gyro_unwrapped
    },
    ekf_data={
        'roll': ekf_roll[1:length],
        'pitch': ekf_pitch[1:length],
        'yaw': ekf_yaw_unwrapped[1:length]
    }
)


# Save complementary Euler outputs
out_comp = pd.DataFrame({
    'timestamp': imu_time[1:length],
    'roll': roll_list,
    'pitch': pitch_list,
    'yaw': yaw_list_unwrapped
})
out_comp.to_csv('result_csv/complementary_euler.csv', index=False)

# Save EKF quats from your ekf_df
ekf_out = pd.DataFrame({
    'timestamp': (ekf_df['%time'] - ekf_df['%time'].min())/1e9,
    'qw': ekf_df['field.orientation.w'],
    'qx': ekf_df['field.orientation.x'],
    'qy': ekf_df['field.orientation.y'],
    'qz': ekf_df['field.orientation.z'],
})



ekf_out.to_csv('result_csv/px4_attitude.csv', index=False)
