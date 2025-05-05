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
        self.build_lookup_tables()

    def build_lookup_tables(self):
        for alpha_deg in range(-50, 51):  # from -50 to 50 inclusive
            idx = alpha_deg
            alpha_rad = np.deg2rad(alpha_deg)
            rim = 2 if alpha_deg < 0 else (3 if alpha_deg == 0 else 4)

            P_poly = self.calculate_P_poly(rim, alpha_rad)
            self.P_poly_table[idx] = P_poly

            P_poly_deriv = np.zeros((2, 7))
            for k in range(7):
                P_poly_deriv[:, k] = P_poly[:, k + 1] * (k + 1)
            self.P_poly_deriv_table[idx] = P_poly_deriv

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
            rot_alpha = self.rotate(alpha + np.pi)
            return rot_alpha @ (H_l - U_l) * scaled_radius + U_l
        elif rim == 2:
            rot_alpha = self.rotate(alpha)
            return rot_alpha @ (G - L_l) * scaled_radius + L_l
        elif rim == 3:
            rot_alpha = self.rotate(alpha)
            return rot_alpha @ (G - L_l) * self.leg_model.r / self.leg_model.R + G
        elif rim == 4:
            rot_alpha = self.rotate(alpha)
            return rot_alpha @ (G - L_r) * scaled_radius + L_r
        elif rim == 5:
            rot_alpha = self.rotate(alpha - np.pi)
            return rot_alpha @ (H_r - U_r) * scaled_radius + U_r
        else:
            return np.zeros((2, 8))
    
    def rotate(self, alpha):
        rot_alpha = np.array([[np.cos(alpha), -np.sin(alpha)],
                              [np.sin(alpha),  np.cos(alpha)]])
        return rot_alpha

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
        tor = np.array([[torque_r], [torque_l]])
        alphas = list(range(-50, 51))
        n = len(alphas)
        positions = np.zeros((2, n))
        forces = np.zeros((2, n))

        # prepare monomials
        phi = np.array([theta**k for k in range(8)])
        phi_deriv = np.array([theta**k for k in range(7)])

        for i, alpha_deg in enumerate(alphas):
            P_poly  = self.P_poly_table[alpha_deg]
            P_deriv = self.P_poly_deriv_table[alpha_deg]

            # compute position via P_poly @ [1, θ, θ^2,...]
            P_theta = P_poly.dot(phi)
            rot_beta = self.rotate(beta)
            pos_world = rot_beta.dot(P_theta)
            positions[:, i] = pos_world

            # derivative
            P_theta_deriv = P_deriv.dot(phi_deriv)

            J = self.calculate_jacobian(P_theta, P_theta_deriv, beta)
            if np.linalg.norm(J) < 1e-6:
                forces[:, i] = 0
            else:
                f = np.linalg.inv(J.T).dot(tor).flatten()
                forces[:, i] = f

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



if __name__ == '__main__':
    leg_model = LegModel(sim=True)
    estimator = ContactEstimator(leg_model)

    theta = 1.93618
    beta = 0.497584

    torque_r = -3.16241
    torque_l = 1.1205

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
    ax = plot_leg.plot_by_angle(theta, beta, O=[0,0], ax=ax)

    # Plot force vectors at each sampled alpha point
    for (x, y), (fx, fy) in zip(positions.T, forces.T):
        ax.arrow(x, y, fx/1000, fy/1000, head_width=0.005, head_length=0.01, length_includes_head=True, color='r')

    ax.set_aspect('equal')
    ax.grid(True)
    plt.show()