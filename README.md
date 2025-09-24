# Attitude-Estimation-with-Madgwick-and-Complementary-Filter-for-UAV-QUAD-

Tuning tips (if you want madgwick's yaw even closer to complementary)

Increase fc_yaw from 2.0 → 3–4 Hz for more mag influence (more pronounced dips/peaks).

If you see jitter, lower fc_yaw or raise gate_frac a bit (e.g., 0.4).

Keep beta around 0.05–0.10 for roll/pitch; raise if you need snappier RP, lower if too noisy.

If you want a perfect match to your complementary yaw, you can literally replace the yaw we compute with your complementary yaw, while keeping Madgwick’s roll/pitch.
