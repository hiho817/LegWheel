import numpy as np
from FittedCoefficient import *
from LegModel import LegModel
import matplotlib.pyplot as plt
from PlotLeg import PlotLeg

class ContactEstimator:
    def __init__(self, leg_model: LegModel):
        self.leg_model = leg_model
        self.P_poly_table = {}
        self.P_poly_deriv_table = {}
        self.build_lookup_tables(step_deg=3.0)

    def build_lookup_tables(self, step_deg: float = 1.0):
        self._step_deg = step_deg
        self._scale    = 1.0 / step_deg
        self.P_poly_table       = {2: {}, 3: {}, 4: {}}
        self.P_poly_deriv_table = {2: {}, 3: {}, 4: {}}

        # 各 rim 的 α 範圍（左閉右閉）
        rim_ranges = {
            2: (-50.0,   -step_deg),
            3: (  0.0, 180.0),
            4: (  step_deg,  50.0),
        }

        for rim, (alpha_min, alpha_max) in rim_ranges.items():
            # np.arange(…+step_deg) 可確保包含上界
            for alpha_deg in np.arange(alpha_min, alpha_max + step_deg, step_deg):
                idx = int(round(alpha_deg * self._scale))
                alpha_rad = np.deg2rad(alpha_deg)

                P_poly = self.calculate_P_poly(rim, alpha_rad)
                self.P_poly_table[rim][idx] = P_poly

                # 一次算好導數
                P_poly_deriv = P_poly[:, 1:] * np.arange(1, 8)   # shape (2,7)
                self.P_poly_deriv_table[rim][idx] = P_poly_deriv
    
    def _idx(self, alpha_deg: float) -> int:
        return int(round(alpha_deg / self._step_deg))
    
    def get_P_poly(self, rim: int, alpha_deg: float):
        return self.P_poly_table[rim][self._idx(alpha_deg)]

    def get_P_poly_deriv(self, rim: int, alpha_deg: float):
        return self.P_poly_deriv_table[rim][self._idx(alpha_deg)]

    def calculate_P_poly(self, rim, alpha):
        H_l = np.vstack((H_x_coef, H_y_coef))
        H_r = np.vstack((-np.array(H_x_coef), H_y_coef))
        U_l = np.vstack((U_x_coef, U_y_coef))
        U_r = np.vstack((-np.array(U_x_coef), U_y_coef))
        L_l = np.vstack((L_x_coef, L_y_coef))
        L_r = np.vstack((-np.array(L_x_coef), L_y_coef))
        G = np.vstack((np.zeros(8), G_y_coef))

        scaled_radius = self.leg_model.radius / self.leg_model.R

        if rim == 1:
            rot_alpha = rotate(alpha + np.pi)
            return rot_alpha @ (H_l - U_l) * scaled_radius + U_l
        elif rim == 2:
            rot_alpha = rotate(alpha)
            return rot_alpha @ (G - L_l) * scaled_radius + L_l
        elif rim == 3:
            rot_alpha = rotate(alpha)
            return rot_alpha @ (G - L_l) * self.leg_model.r / self.leg_model.R + G
        elif rim == 4:
            rot_alpha = rotate(alpha)
            return rot_alpha @ (G - L_r) * scaled_radius + L_r
        elif rim == 5:
            rot_alpha = rotate(alpha - np.pi)
            return rot_alpha @ (H_r - U_r) * scaled_radius + U_r
        else:
            return np.zeros((2, 8))

    def calculate_jacobian(self, P_theta, P_theta_deriv, beta):
        cos_b, sin_b = np.cos(beta), np.sin(beta)
        dtheta_dphiR, dtheta_dphiL = -0.5, 0.5
        dbeta_dphiR, dbeta_dphiL = 0.5, 0.5

        dPx_dtheta = P_theta_deriv[0] * cos_b - P_theta_deriv[1] * sin_b
        dPy_dtheta = P_theta_deriv[0] * sin_b + P_theta_deriv[1] * cos_b
        dPx_dbeta = P_theta[0] * (-sin_b) - P_theta[1] * cos_b
        dPy_dbeta = P_theta[0] * cos_b + P_theta[1] * (-sin_b)

        J = np.array([
            [dPx_dtheta * dtheta_dphiR + dPx_dbeta * dbeta_dphiR,
             dPx_dtheta * dtheta_dphiL + dPx_dbeta * dbeta_dphiL],
            [dPy_dtheta * dtheta_dphiR + dPy_dbeta * dbeta_dphiR,
             dPy_dtheta * dtheta_dphiL + dPy_dbeta * dbeta_dphiL]
        ])
        return J

    def sample_force_and_positions(self, theta, beta, torque_r, torque_l):

        self.calculate_outside_point(theta, beta)
        G_alpha = max(0.0, min(self.get_G_alpha_interval(), 180.0))
        # alpha_bias = int(self.get_L_rim_alpha_interval())

        step   = self._step_deg
        scale  = self._scale  # = 1/step

        alpha_tasks = []  # list[(rim, α_deg)]
        
        # rim 2: [-50, 0]
        for alpha in np.arange(-50.0, 0.0, step):
            alpha_tasks.append((2, alpha))
        # rim 3: [0, G_alpha]
        for alpha in np.arange(0.0,  G_alpha + step, step * 10):
            alpha_tasks.append((3, alpha))
        # rim 4: [0, 50]
        for alpha in np.arange(step,  50.0 + step, step):
            alpha_tasks.append((4, alpha))

        n_pts = len(alpha_tasks)
        positions = np.zeros((2, n_pts))
        forces    = np.zeros((2, n_pts))

        # ---------- 3) 事先把 θ 多項式 & 導數算好 ----------
        phi       = np.array([theta ** k for k in range(8)])   # 0~7 次
        phi_deriv = np.array([theta ** k for k in range(7)])   # 0~6 次
        tor       = np.array([[torque_r], [torque_l]])
        rot_beta  = rotate(beta)

        # ---------- 4) 逐點處理 ----------
        for i, (rim, alpha_deg) in enumerate(alpha_tasks):
            idx = int(round(alpha_deg * scale))        # == self._idx(α_deg)

            P_poly  = self.P_poly_table[rim][idx]
            P_deriv = self.P_poly_deriv_table[rim][idx]

            # --- 位置 ---
            P_theta  = P_poly @ phi            # shape (2,)
            pos_world = rot_beta @ P_theta
            positions[:, i] = pos_world

            # --- 力 ---
            P_theta_d = P_deriv @ phi_deriv    # shape (2,)
            J         = self.calculate_jacobian(P_theta, P_theta_d, beta)

            if np.linalg.cond(J) > 1e12:   # 奇異或接近奇異
                forces[:, i] = 0.0
            else:
                f = np.linalg.inv(J.T) @ tor
                forces[:, i] = f.flatten()

        return positions, forces

    
    def estimate_force(self, theta, beta, torque_r, torque_l):
        self.leg_model.contact_map(theta, beta)
        rim = self.leg_model.rim
        alpha = self.leg_model.alpha

        P_poly = self.calculate_P_poly(rim, alpha)

        # Compute polynomial terms
        P_theta = sum([P_poly[:, i] * theta**i for i in range(8)])
        P_poly_deriv = np.array([P_poly[:, i+1] * (i+1) for i in range(7)]).T
        P_theta_deriv = sum([P_poly_deriv[:, i] * theta**i for i in range(7)])

        jacobian = self.calculate_jacobian(P_theta, P_theta_deriv, beta)

        if np.linalg.norm(jacobian) < 1e-6:
            return np.zeros((2, 1))
        else:
            torque = np.array([[torque_r], [torque_l]])
            force_est = np.linalg.inv(jacobian.T) @ torque
            return force_est
        
    def get_G_alpha_interval(self):
        z1 = self.LG_r - self.leg_model.G
        z2 = self.LG_l - self.leg_model.G
        angle = np.angle(z1 / z2)
        angle = np.rad2deg(angle)
        return angle
    
    # def get_L_rim_alpha_interval(self):
    #     z1 = self.leg_model.U_l - self.leg_model.L_l
    #     z2 = self.leg_model.F_l - self.leg_model.L_l
    #     angle_1 = abs(np.angle(z1 / z2))
    #     print("angle_1:", angle_1)

    #     a = abs(z1)
    #     b = abs(self.LF_l - self.leg_model.L_l)
    #     c = abs(self.UF_l - self.leg_model.U_l)
    #     print("a:", a)
    #     angle_2 = np.angle(np.arccos((a**2 + b**2 - c**2) / (2 * a * b)))
    #     print("angle_2:", angle_2)
    #     angle = abs(angle_1 - angle_2)
        
    #     print("L_rim_alpha_interval:", angle)
    #     return angle

    
    def calculate_outside_point(self,theta, beta):
        self.leg_model.forward(theta, beta, vector=False)
        self.LG_l = (self.leg_model.G - self.leg_model.L_l)   / self.leg_model.R * self.leg_model.radius + self.leg_model.L_l   # L_l -> G -> rim point
        self.LG_r = (self.leg_model.G - self.leg_model.L_r)   / self.leg_model.R * self.leg_model.radius + self.leg_model.L_r   # L_r -> G -> rim point
        self.UH_l = (self.leg_model.H_l - self.leg_model.U_l) / self.leg_model.R * self.leg_model.radius + self.leg_model.U_l   # U_l -> H_l -> rim point
        self.UH_r = (self.leg_model.H_r - self.leg_model.U_r) / self.leg_model.R * self.leg_model.radius + self.leg_model.U_r   # U_r -> H_r -> rim point
        self.LF_l = (self.leg_model.F_l - self.leg_model.L_l) / self.leg_model.R * self.leg_model.radius + self.leg_model.L_l   # L_l -> F_l -> rim point
        self.LF_r = (self.leg_model.F_r - self.leg_model.L_r) / self.leg_model.R * self.leg_model.radius + self.leg_model.L_r   # L_r -> F_r -> rim point
        self.UF_l = (self.leg_model.F_l - self.leg_model.U_l) / self.leg_model.R * self.leg_model.radius + self.leg_model.U_l   # U_l -> F_l -> rim point
        self.UF_r = (self.leg_model.F_r - self.leg_model.U_r) / self.leg_model.R * self.leg_model.radius + self.leg_model.U_r   # U_r -> F_r -> rim point

def rotate(alpha):
    rot_alpha = np.array([[np.cos(alpha), -np.sin(alpha)],
                            [np.sin(alpha),  np.cos(alpha)]])
    return rot_alpha


def plot_complex_point(complex_number, ax=None, label=None):
    if ax is None:
        fig, ax = plt.subplots()
    marker_styles = ['o', 's', 'D', '^', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '|', '_']
    marker = np.random.choice(marker_styles)
    ax.plot(complex_number.real, complex_number.imag, marker=marker, label=label)
    if label:
        ax.legend()
    ax.set_aspect('equal')

if __name__ == '__main__':
    leg_model = LegModel(sim=True)
    estimator = ContactEstimator(leg_model)

    theta = 1.93618
    beta = 0.497584
    torque_r = -3.16241
    torque_l = 1.1205

    # theta = np.deg2rad(30.0)
    # beta = 0.0
    # torque_r = 0.0
    # torque_l = 0.0
    
    # force = estimator.estimate_force(theta, beta, torque_r, torque_l)
    # print ("theta: %f degree" % np.rad2deg(theta))
    # print ("beta: %f degree" % np.rad2deg(beta))
    # print("rim:", leg_model.rim)
    # print("alpha:", leg_model.alpha)
    # print("Estimated Force X:", force[0])
    # print("Estimated Force Y:", force[1])

    positions, forces = estimator.sample_force_and_positions(theta, beta, torque_r, torque_l)
    # print("Sampled force shape:", forces.shape)  # (2, 101)

    plot_leg = PlotLeg(sim=True)
    fig, ax = plt.subplots(figsize=(6,6))
    # ax = plot_leg.plot_by_angle(theta, beta, O=[0,0], ax=ax)

    # # Plot force vectors at each sampled alpha point
    # for (x, y), (fx, fy) in zip(positions.T, forces.T):
    #     ax.arrow(x, y, (-fx)/1000, (-fy + 0.68*9.81)/1000, head_width=0.005, head_length=0.01, length_includes_head=True, color='r')
    # #                                 #wheel weight = 0.68*9.81
                            
    # # Draw the points
    # plot_complex_point(estimator.LG_l, ax = ax, label='LG_l')
    # plot_complex_point(estimator.LF_l, ax = ax, label='LF_l')
    # plot_complex_point(estimator.UF_l, ax = ax, label='UF_l')
    # plot_complex_point(estimator.leg_model.L_l, ax = ax, label='L_l')


    ax.grid(True)
    plt.show()