# Complementary vs. Madgwick on Pixhawk IMU

This project is a hands-on exploration of attitude estimation (roll, pitch, yaw) from raw IMU + magnetometer data, and a comparison of simple filters against PX4’s onboard EKF.

The focus is on learning: calibrating sensors, implementing filters step-by-step, and visualizing how they stack up to PX4’s “ground truth.”

---
( mag calibration is still in progress to improve thr yaw errors )
<img width="1189" height="989" alt="T_15_40_secs" src="https://github.com/user-attachments/assets/bde1cb0d-afdb-459c-8fd8-0e5d76fc9812" />


## What this project does

1. **Magnetometer Calibration**
   Corrects hard-iron (offset) and soft-iron (scaling) errors using sphere fitting and covariance whitening.

2. **Complementary Filter**
   Blends gyro integration (smooth but drifting) with accel/mag (noisy but absolute). Optional accel gating reduces trust during high linear acceleration.

3. **Madgwick Filter**
   6-DoF update with yaw-only mag correction. Keeps roll/pitch from accel, uses mag to correct heading drift.

4. **PX4 EKF Reference**
   Uses PX4’s onboard attitude quaternions as reference, interpolated to IMU time.

5. **Comparison**
   Overlay plots and metrics (MAE, RMSE, quaternion error) to see how Complementary and Madgwick compare with PX4.

---

## Repository structure

* `mag_calibration.py` — Magnetometer calibration.
* `complementary_filter.py` — Complementary filter (IMU + mag).
* `madgwick_filter.py` — Madgwick filter (IMU + yaw mag correction).
* `compare.py` — Aligns Complementary & Madgwick with PX4 EKF and computes metrics.
* `plots/` — Auto-saved figures + metrics.
* Example CSVs:

  * `main_bag_imu.csv`
  * `main_bag_mag.csv`
  * `main_bag_ekf.csv`

---

## How to run

You can follow either the **notebook (Colab)** path or the **scripted** path.

### A) Notebook (Colab) workflow

1. Run the **Complementary** cell → defines `imu_df`, `mag_df`, `imu_time`, and arrays (`roll_list`, `pitch_list`, `yaw_list_unwrapped`).
2. Run the **Madgwick** cell → auto-uses calibrated mag if available, computes Madgwick, and overlays directly against the in-memory complementary arrays.

   * No CSVs are needed for the overlay.
   * To save the figure, add:

     ```python
     plt.savefig("overlay_madgwick_vs_complementary.png", dpi=150, bbox_inches="tight")
     ```
3. Optional: both filters also save their own CSVs (`complementary_euler.csv`, `madgwick_euler.csv`).

### B) Scripted (repo) workflow

1. `python mag_calibrate.py` → writes `mag_cal.json` (+ calibrated CSV).
2. `python complementary.py` → writes `complementary_euler.csv`.
3. `python madgwick.py` → writes `madgwick_euler.csv`.
4. `python compare.py` → aligns both with PX4 EKF and saves metrics/plots under `plots/`.

---

## Tuning parameters

All scripts include a few knobs you can adjust depending on your dataset:

* **Complementary filter**

  * `FC_ROLL_PITCH` (default: 5–10 Hz)
    Higher → more trust in accel for roll/pitch.
  * `FC_YAW` (default: 1–3 Hz)
    Lower → smoother yaw (less mag noise).
  * `USE_ACCEL_GATING` (True/False)
    If enabled, accel input is ignored when total accel ≠ gravity.
  * `SEGMENT_START_SEC`, `SEGMENT_END_SEC`
    Time window for plotting. Set both to `None` for full run.

* **Madgwick filter**

  * `beta` (default: 0.05–0.1)
    Higher → faster correction (less drift), but noisier.
  * `fc_yaw` (default: ~2 Hz)
    Controls blending of mag into yaw; lower = smoother.
  * `gate_frac` (default: 0.3–0.4)
    Rejects mag outliers when |m| deviates too much from median norm.

* **Comparison**

  * `ENABLE_TIME_SHIFT` (True/False)
    Enables auto time-shift search between PX4 and your filter.
  * `REMOVE_CONSTANT_YAW_BIAS` (True/False)
    Removes constant heading offset before computing errors.
  * `LAG_RANGE` (default: ±0.4 s)
    Search window for time shift alignment.

---

## Key takeaways

* Calibrated mag is critical for stable yaw.
* Complementary filter is simple and decent, but drifts with gyro bias.
* Madgwick with yaw-only mag correction was the most stable in tests.
* PX4’s EKF remains smoother since it fuses more sensors, but our DIY filters get close for attitude.

---

## Why I built this

I wanted to peek under the hood of PX4’s EKF and rebuild simpler filters line by line. This way I can see the math in action, not just trust a black box.

---

## Next steps

* Extend toward a minimal EKF implementation.
* Add barometer/GPS for position/velocity.
* Test with live ROS2 topics, not just offline CSVs.
* Wrap filters into reusable ROS2 nodes.
