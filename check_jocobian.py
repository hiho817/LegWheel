import numpy as np
from FittedCoefficient import *
from LegModel import LegModel
import matplotlib.pyplot as plt
from PlotLeg import PlotLeg

class ContactEstimator:
    def __init__(self, leg_model: LegModel, theta, beta):
        self.leg_model = leg_model
        self.theta = theta
        self.beta = beta
        self.P_poly_table = {}
        self.P_poly_deriv_table = {}
        self.build_lookup_tables(step_deg=0.1)

    def build_lookup_tables(self, step_deg: float = 1.0):
        self._step_deg = step_deg
        self._scale    = 1.0 / step_deg
        self.P_poly_table       = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
        self.P_poly_deriv_table = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}

        # 各 rim 的 α 範圍（左閉右閉）
        rim_ranges = {
            1: (-180.0, -50.0),
            2: (-50.0,   -step_deg),
            3: (  0.0, 180.0),
            4: (  step_deg,  50.0),
            5: (50.0 + step_deg, 180.0),
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
    
    def get_G_alpha_interval(self):
        z1 = self.LG_r - self.leg_model.G
        z2 = self.LG_l - self.leg_model.G
        angle = np.angle(z1 / z2)
        angle = np.rad2deg(angle)
        return angle
    
    def calculate_outside_point(self):
        self.leg_model.forward(self.theta, self.beta, vector=False)
        self.LG_l = (self.leg_model.G - self.leg_model.L_l)   / self.leg_model.R * self.leg_model.radius + self.leg_model.L_l   # L_l -> G -> rim point
        self.LG_r = (self.leg_model.G - self.leg_model.L_r)   / self.leg_model.R * self.leg_model.radius + self.leg_model.L_r   # L_r -> G -> rim point
        self.UH_l = (self.leg_model.H_l - self.leg_model.U_l) / self.leg_model.R * self.leg_model.radius + self.leg_model.U_l   # U_l -> H_l -> rim point
        self.UH_r = (self.leg_model.H_r - self.leg_model.U_r) / self.leg_model.R * self.leg_model.radius + self.leg_model.U_r   # U_r -> H_r -> rim point
        self.LF_l = (self.leg_model.F_l - self.leg_model.L_l) / self.leg_model.R * self.leg_model.radius + self.leg_model.L_l   # L_l -> F_l -> rim point
        self.LF_r = (self.leg_model.F_r - self.leg_model.L_r) / self.leg_model.R * self.leg_model.radius + self.leg_model.L_r   # L_r -> F_r -> rim point
        self.UF_l = (self.leg_model.F_l - self.leg_model.U_l) / self.leg_model.R * self.leg_model.radius + self.leg_model.U_l   # U_l -> F_l -> rim point
        self.UF_r = (self.leg_model.F_r - self.leg_model.U_r) / self.leg_model.R * self.leg_model.radius + self.leg_model.U_r   # U_r -> F_r -> rim point

    def analyze_jacobian_vs_alpha(self):
        """
        分析在固定 theta, beta 下,Jacobian 四個元素隨 alpha 的變化
        包含所有 5 個 rim 段
        """
        self.calculate_outside_point()
        G_alpha = max(0.0, min(self.get_G_alpha_interval(), 180.0))

        alpha_tasks = []  # list[(rim, α_deg)]

        step = self._step_deg

        # rim 1: [-180, -50]
        for alpha in np.arange(-180.0, -50.0 + step, step):
            alpha_tasks.append((1, alpha))
        # rim 2: [-50, 0]
        for alpha in np.arange(-50.0, 0.0, step):
            alpha_tasks.append((2, alpha))
        # rim 3: [0, G_alpha]
        for alpha in np.arange(0.0,  G_alpha + step, step):
            alpha_tasks.append((3, alpha))
        # rim 4: [0, 50]
        for alpha in np.arange(step,  50.0 + step, step):
            alpha_tasks.append((4, alpha))
        # rim 5: [50, 180]
        for alpha in np.arange(50.0 + step, 180.0 + step, step):
            alpha_tasks.append((5, alpha))
        
        n_pts = len(alpha_tasks)
        
        # 儲存結果
        all_alpha_values = []
        all_jacobian_elements = np.zeros((4, n_pts))  # J[0,0], J[0,1], J[1,0], J[1,1]
        all_rim_labels = []

        phi = np.array([self.theta ** k for k in range(8)])   # 0~7 次
        phi_deriv = np.array([self.theta ** k for k in range(7)])   # 0~6 次

        for i, (rim, alpha_deg) in enumerate(alpha_tasks):
            try:
                idx = self._idx(alpha_deg)
                
                # 檢查查找表中是否有對應值
                if rim in self.P_poly_table and idx in self.P_poly_table[rim]:
                    P_poly = self.P_poly_table[rim][idx]
                    P_deriv = self.P_poly_deriv_table[rim][idx]
                else:
                    # 直接計算
                    alpha_rad = np.deg2rad(alpha_deg)
                    P_poly = self.calculate_P_poly(rim, alpha_rad)
                    P_deriv = P_poly[:, 1:] * np.arange(1, 8)

                P_theta = P_poly @ phi 
                P_theta_d = P_deriv @ phi_deriv

                J = self.calculate_jacobian(P_theta, P_theta_d, self.beta)
                
                # 儲存四個 Jacobian 元素
                all_jacobian_elements[0, i] = J[0, 0]  # dPx/dphiR
                all_jacobian_elements[1, i] = J[0, 1]  # dPx/dphiL
                all_jacobian_elements[2, i] = J[1, 0]  # dPy/dphiR
                all_jacobian_elements[3, i] = J[1, 1]  # dPy/dphiL
                
            except (KeyError, IndexError, ValueError) as e:
                # 如果出錯，設為 NaN
                all_jacobian_elements[:, i] = np.nan
                print(f"Error at rim {rim}, alpha {alpha_deg}: {e}")
            
            all_alpha_values.append(alpha_deg)
            all_rim_labels.append(str(rim))
        
        return np.array(all_alpha_values), all_jacobian_elements, np.array(all_rim_labels)
    
    def plot_jacobian_vs_alpha(self):
        """
        繪製 Jacobian 四個元素隨 alpha 的變化圖
        """
        alpha_values, jacobian_elements, rim_labels = self.analyze_jacobian_vs_alpha()
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'Jacobian Elements vs Alpha\n(θ={np.rad2deg(self.theta):.1f}°, β={np.rad2deg(self.beta):.1f}°)')
        
        element_names = ['J[0,0] (dPx/dφR)', 'J[0,1] (dPx/dφL)', 
                        'J[1,0] (dPy/dφR)', 'J[1,1] (dPy/dφL)']
        
        # 為不同 rim 段用不同顏色
        rim_colors = {
            '1': 'orange',
            '2': 'red',
            '3': 'blue', 
            '4': 'green',
            '5': 'purple'
        }
        
        for i, (ax, name) in enumerate(zip(axes.flat, element_names)):
            # 為每個 rim 段繪製不同顏色的線
            for rim_label, color in rim_colors.items():
                mask = rim_labels == rim_label
                if np.any(mask):
                    alpha_subset = alpha_values[mask]
                    jacobian_subset = jacobian_elements[i][mask]
                    
                    # 移除 NaN 值
                    valid_mask = ~np.isnan(jacobian_subset)
                    if np.any(valid_mask):
                        ax.plot(alpha_subset[valid_mask], jacobian_subset[valid_mask], 
                               color=color, linewidth=1.5, label=f'Rim {rim_label}', marker='.')
            
            ax.set_xlabel('Alpha (degrees)')
            ax.set_ylabel('Jacobian Element')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        plt.show()
        
        return alpha_values, jacobian_elements

def rotate(alpha):
    rot_alpha = np.array([[np.cos(alpha), -np.sin(alpha)],
                            [np.sin(alpha),  np.cos(alpha)]])
    return rot_alpha

if __name__ == '__main__':
    leg_model = LegModel(sim=True)

    theta = np.deg2rad(50.0)
    beta = 0.0
    
    estimator = ContactEstimator(leg_model, theta, beta)

    plot_leg = PlotLeg(sim=True)
    ax = plot_leg.plot_by_angle(theta, beta, O=[0,0])
    
    # 分析 Jacobian 隨 alpha 的變化
    alpha_values, jacobian_elements, rim_labels = estimator.analyze_jacobian_vs_alpha()
    
    # 繪製結果
    estimator.plot_jacobian_vs_alpha()