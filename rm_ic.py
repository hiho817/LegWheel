import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Coefficients for 4th order polynomial (theta^4, theta^3, theta^2, theta^1, theta^0)
rm_coeff = np.array([-0.0035, 0.0110, 0.0030, 0.0500, -0.0132])
Io_coeff = np.array([0.0001, -0.0001, -0.0013, 0.0043, 0.0041])
# Ic_coeff = np.array([0.0001, -0.0003, -0.0014, 0.0049, 0.0047])
# Ic_coeff = np.array([7*10**-5 , -0.0001, -0.0013, 0.0043, 0.0041])
Ic_coeff = np.array([1e-06, -1E-05, 0.0001, -0.0002, -0.0012, 0.0042, 0.0041])

# Create theta range from 17 to 160 degrees
theta_deg = np.linspace(17, 150, 200)
theta_rad = np.deg2rad(theta_deg)

# Evaluate polynomials

m = 0.68

rm_values = np.polyval(rm_coeff, theta_rad)
Ic_values = np.polyval(Ic_coeff, theta_rad)
Io_values_origin = np.polyval(Io_coeff, theta_rad)
parallel = (m * rm_values * rm_values)
# Ic_values = Io_values - parallel
# Io_values_parallel = Ic_values + parallel


# Use matplotlib's built-in math rendering instead of LaTeX
plt.rcParams.update({
    "text.usetex": False,  # Disable LaTeX
    "mathtext.fontset": "cm",  # Use Computer Modern font for math
    "font.family": "serif",
    "font.size": 50,
    "axes.labelsize": 30,
    "axes.titlesize": 30,
})

# Create separate figures instead of one figure with subplots
# Plot Rm vs theta
fig_rm = plt.figure(figsize=(7, 7))  # Changed to square dimensions
ax1 = fig_rm.add_subplot(111)
ax1.plot(theta_rad, rm_values, 'b-', linewidth=2.5)
ax1.set_xlabel(r'$\theta$ (rad)')
ax1.set_ylabel(r'$R_m$ (m)')
# ax1.set_title(r'$R_m$ vs. $\theta$')
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.grid(True)
fig_rm.tight_layout()
fig_rm.savefig('Rm_vs_Theta.png', dpi=300, bbox_inches='tight')  # Save Rm figure as PDF

# Plot Ic vs theta
fig_ic = plt.figure(figsize=(7, 7))  # Changed to square dimensions
ax2 = fig_ic.add_subplot(111)
ax2.plot(theta_rad, Ic_values, 'b-', linewidth=2.5)
ax2.set_xlabel(r'$\theta$ (rad)')
ax2.set_ylabel(r'$I_c$(kg$\cdot$m$^2$)')
# ax2.set_title(r'$I_c$ vs. $\theta$')
ax2.tick_params(axis='both', which='major', labelsize=20)
ax2.grid(True)
fig_ic.tight_layout()
fig_ic.savefig('Ic_vs_Theta.png', dpi=300, bbox_inches='tight')  # Save Ic figure as PDF

# Show both figures
plt.show()