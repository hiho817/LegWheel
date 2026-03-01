import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import LegModel
import PlotLeg

"""
Generate theta, beta CSV for a single leg moving at 0.1 m/s.
Only one step's contact (stance) phase is used.
Following WalkTrajectory.py approach.
"""

sim = False
leg_model = LegModel.LegModel(sim=sim)

#### User-defined parameters ####
velocity = 0.1          # hip velocity, m/s
sampling = 1000         # sampling rate, commands per second
stand_height = 0.25     # standing height, m
step_length = 0.3       # step length, m
swing_time = 0.2        # swing time ratio (duty: 0.8~1.0)

#### Dependent parameters ####
dS = velocity / sampling                              # hip displacement per sample
incre_duty = dS / step_length                          # duty increment per sample

# Single leg: hip starts at origin, height = stand_height
hip = np.array([0.0, stand_height])

# Initial foothold: foot is ahead of hip at start of contact phase
# (same as WalkTrajectory.py leg index 3 pattern: foot forward at contact start)
contact_rim_names = ["G", "L_l", "L_r", "U_l", "U_r"]
rim_idx = [3, 2, 4, 1, 5]
contact_height = [leg_model.r, leg_model.radius, leg_model.radius, leg_model.radius, leg_model.radius]

foothold = hip + np.array([step_length / 2 * (1 - swing_time), -stand_height])

#### Initial theta, beta (following WalkTrajectory.py) ####
for j in range(5):
    theta, beta = leg_model.inverse(
        foothold + np.array([0, contact_height[j]]) - hip, contact_rim_names[j]
    )
    leg_model.contact_map(theta, beta)
    if leg_model.rim == rim_idx[j]:
        break

#### Contact phase simulation ####
theta_list = [theta]
beta_list = [beta]
hip_list = [hip.copy()]

duty = 0.0  # start of stance phase

while duty < (1 - swing_time):
    # stance phase: use leg_model.move with hip displacement
    new_hip = hip_list[-1] + np.array([dS, 0.0])
    move_vec = new_hip - hip_list[-1]
    theta, beta = leg_model.move(theta_list[-1], beta_list[-1], move_vec)
    theta_list.append(theta)
    beta_list.append(beta)
    hip_list.append(new_hip.copy())
    duty += incre_duty

theta_arr = np.array(theta_list)
beta_arr = np.array(beta_list)
hip_arr = np.array(hip_list)
num_samples = len(theta_arr)

#### Compute time and hip velocity ####
dt = 1.0 / sampling
time_arr = np.arange(num_samples) * dt  # time starting from 0 s
# Hip velocity: X = rightward, Z = upward
hip_vx = np.gradient(hip_arr[:, 0], dt)  # d(hip_x)/dt
hip_vz = np.gradient(hip_arr[:, 1], dt)  # d(hip_z)/dt

#### Save to CSV ####
df = pd.DataFrame({
    "time": time_arr,
    "theta": theta_arr,
    "beta": beta_arr,
    "hip_vx": hip_vx,
    "hip_vz": hip_vz,
})
output_name = "single_leg_contact_theta_beta.csv"
df.to_csv(output_name, index=False)

contact_duration = num_samples / sampling
print(f"Saved {num_samples} samples to {output_name}")
print(f"  Velocity:         {velocity} m/s")
print(f"  Sampling rate:    {sampling} Hz")
print(f"  Contact duration: {contact_duration} s")
print(f"  Contact distance: {contact_duration * velocity:.4f} m")
print(f"  Theta range:      [{np.rad2deg(theta_arr.min()):.2f}, {np.rad2deg(theta_arr.max()):.2f}] deg")
print(f"  Beta range:       [{np.rad2deg(beta_arr.min()):.2f}, {np.rad2deg(beta_arr.max()):.2f}] deg")
print(f"  Hip Vx range:     [{hip_vx.min():.4f}, {hip_vx.max():.4f}] m/s")
print(f"  Hip Vz range:     [{hip_vz.min():.4f}, {hip_vz.max():.4f}] m/s")

#### Animation for validation ####
fps = 30
divide = max(1, num_samples // (fps * int(contact_duration + 1)))

fig, ax = plt.subplots(figsize=(10, 5))
plot_leg = PlotLeg.PlotLeg(sim=sim)
plot_leg.setting()

num_frames = num_samples // divide

def plot_update(frame):
    ax.clear()
    ax.set_aspect('equal')
    idx = frame * divide

    hx = hip_arr[idx, 0]
    ax.set_xlim(hx - 0.35, hx + 0.35)
    ax.set_ylim(-0.05, 0.45)

    # Ground
    ax.plot([hx - 0.4, hx + 0.4], [0, 0], 'g-', linewidth=2)

    # Hip marker (orange cross)
    ax.plot(hip_arr[idx, 0], hip_arr[idx, 1], '+', color='orange', ms=14, mew=3, zorder=5)

    # Draw leg
    plot_leg.plot_leg(theta_arr[idx], beta_arr[idx], hip_arr[idx], ax)

    ax.set_title(f'Contact Phase  |  t = {idx / sampling:.3f} s  |  '
                 f'θ = {np.rad2deg(theta_arr[idx]):.1f}°  β = {np.rad2deg(beta_arr[idx]):.1f}°')
    ax.grid(True)

ani = FuncAnimation(fig, plot_update, frames=num_frames, interval=1000 // fps)
ani.save("single_leg_contact_animation.gif", writer='pillow', fps=fps)
print(f"Animation saved to single_leg_contact_animation.gif ({num_frames} frames)")
plt.show()
