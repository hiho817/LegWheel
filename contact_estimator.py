import numpy as np
from FittedCoefficient import *
from LegModel import LegModel

class ForceEstimator:
    def __init__(self, leg_model: LegModel):
        self.leg_model = leg_model
        self.R = leg_model.R
        self.r = leg_model.r
        self.radius = leg_model.radius

    def calculate_P_poly(self, rim, alpha):
        # build rim-specific coefficient matrices
        H_l = np.vstack((H_x_coef, H_y_coef))
        H_r = np.vstack((-np.array(H_x_coef), H_y_coef))
        U_l = np.vstack((U_x_coef, U_y_coef))
        U_r = np.vstack((-np.array(U_x_coef), U_y_coef))
        L_l = np.vstack((L_x_coef, L_y_coef))
        L_r = np.vstack((-np.array(L_x_coef), L_y_coef))
        F_l = np.vstack((F_x_coef, F_y_coef))
        F_r = np.vstack((-np.array(F_x_coef), F_y_coef))
        G = np.vstack((np.zeros(8), G_y_coef))

        scaled_radius = self.radius / self.R
        rot_alpha = np.array([[np.cos(alpha), -np.sin(alpha)],
                              [np.sin(alpha),  np.cos(alpha)]])

        if rim == 1:
            P_poly = rot_alpha @ (H_l - U_l) * scaled_radius + U_l
        elif rim == 2:
            P_poly = rot_alpha @ (G - L_l) * scaled_radius + L_l
        elif rim == 3:
            P_poly = rot_alpha @ (G - L_l) * self.r / self.R + G
        elif rim == 4:
            P_poly = rot_alpha @ (G - L_r) * scaled_radius + L_r
        elif rim == 5:
            rot_alpha = np.array([[np.cos(alpha - np.pi), -np.sin(alpha - np.pi)],
                                  [np.sin(alpha - np.pi),  np.cos(alpha - np.pi)]])
            P_poly = rot_alpha @ (H_r - U_r) * scaled_radius + U_r
        else:
            P_poly = np.zeros((2, 8))
        return P_poly

    def calculate_jacobian(self, P_theta, P_theta_deriv, beta):
        cos_b = np.cos(beta)
        sin_b = np.sin(beta)

        dtheta_dphiR, dtheta_dphiL = -0.5, 0.5
        dbeta_dphiR, dbeta_dphiL = 0.5, 0.5

        dPx_dtheta = P_theta_deriv[0] * cos_b - P_theta_deriv[1] * sin_b
        dPy_dtheta = P_theta_deriv[0] * sin_b + P_theta_deriv[1] * cos_b
        dPx_dbeta = -P_theta[0] * sin_b - P_theta[1] * cos_b
        dPy_dbeta = P_theta[0] * cos_b - P_theta[1] * sin_b

        J = np.array([
            [dPx_dtheta * dtheta_dphiR + dPx_dbeta * dbeta_dphiR,
             dPx_dtheta * dtheta_dphiL + dPx_dbeta * dbeta_dphiL],
            [dPy_dtheta * dtheta_dphiR + dPy_dbeta * dbeta_dphiR,
             dPy_dtheta * dtheta_dphiL + dPy_dbeta * dbeta_dphiL],
        ])
        return J

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
    estimator = ForceEstimator(leg_model)

    theta = np.deg2rad(30)
    beta = np.deg2rad(25)
    torque_r = 2.5
    torque_l = 2.5

    force = estimator.estimate_force(theta, beta, torque_r, torque_l)
    print("Estimated Force:", force)