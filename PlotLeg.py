import numpy as np
from LegModel import *
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
from Contact_Map import *
from FittedCoefficient import *

class LegAnimation(LegModel):
    def __init__(self):
        super().__init__()
        self.forward(np.deg2rad(17), 0, vector=False)
        self.O = np.array([0, 0])   # origin of leg in world coordinate
        self.contact_map = ContactMap()

        self.leg_shape = self.LegShape(self, self.O)   # initial pose of leg
   
    
    ## Get Shape Of Leg ##  
    class LegShape:
        def __init__(self, linkleg, O):
            self.linkleg = linkleg
            self.R = linkleg.R
            self.r = linkleg.r
            self.outer_radius = self.R + self.r
            self.O = O
            self.fig_size = 10
            self.mark_size = 2.0
            self.line_width = 1.0
            self.get_shape()
            
        class rim:
            def __init__(self, arc, arc_out, start):
                self.arc = [arc, arc_out]   # inner & outer arcs
                self.start = start          # start angle 
                
        def get_shape(self):
            # four rims (inner arc, outer arc, start angle)
            self.upper_rim_r = self.rim( *self.get_arc(self.linkleg.F_r, self.linkleg.H_r, self.linkleg.U_r, 'black', self.r))
            self.upper_rim_l = self.rim( *self.get_arc(self.linkleg.H_l, self.linkleg.F_l, self.linkleg.U_l, 'black', self.r))
            self.lower_rim_r = self.rim( *self.get_arc(self.linkleg.G, self.linkleg.F_r, self.linkleg.L_r, 'black', self.r))
            self.lower_rim_l = self.rim( *self.get_arc(self.linkleg.F_l, self.linkleg.G, self.linkleg.L_l, 'black', self.r))
            # five joints on the rims   (center, radius)
            self.upper_joint_r = self.get_circle(self.linkleg.H_r, self.r) 
            self.upper_joint_l = self.get_circle(self.linkleg.H_l, self.r) 
            self.lower_joint_r = self.get_circle(self.linkleg.F_r, self.r) 
            self.lower_joint_l = self.get_circle(self.linkleg.F_l, self.r) 
            self.G_joint       = self.get_circle(self.linkleg.G, self.r)
            # six bars  (point1, point2)
            self.OB_bar_r = self.get_line(0, self.linkleg.B_r) 
            self.OB_bar_l = self.get_line(0, self.linkleg.B_l) 
            self.AE_bar_r = self.get_line(self.linkleg.A_r, self.linkleg.E)
            self.AE_bar_l = self.get_line(self.linkleg.A_l, self.linkleg.E)
            self.CD_bar_r = self.get_line(self.linkleg.C_r, self.linkleg.D_r)
            self.CD_bar_l = self.get_line(self.linkleg.C_l, self.linkleg.D_l) 
            
        def get_arc(self, p1, p2, o, color='black', offset=0.01):
            start = np.angle(p1-o, deg=True)
            end = np.angle(p2-o, deg=True)
            radius = np.abs(p1-o)
            arc = Arc([o.real, o.imag], 2*(radius-offset), 2*(radius-offset), angle=0.0, theta1=start, theta2=end, color=color, linewidth=self.line_width)
            arc_out = Arc([o.real, o.imag], 2*(radius+offset), 2*(radius+offset), angle=0.0, theta1=start, theta2=end, color=color, linewidth=self.line_width)
            return arc, arc_out, start

        def get_circle(self, o, r, color='black'):
            circle = Arc([o.real, o.imag], 2*r, 2*r, angle=0.0, theta1=0, theta2=360, color=color, linewidth=self.line_width)
            return circle

        def get_line(self, p1, p2, color='black'):
            line = Line2D([p1.real, p2.real], [p1.imag, p2.imag], marker='o', markersize=self.mark_size, linestyle='-', color=color, linewidth=self.line_width)
            return line
        
        ## Set Postion Of Leg ##  
        def update(self):
            # four rims (rim, start point, center)
            self.set_rim(self.upper_rim_r, self.linkleg.F_r, self.linkleg.U_r)
            self.set_rim(self.upper_rim_l, self.linkleg.H_l, self.linkleg.U_l)
            self.set_rim(self.lower_rim_r, self.linkleg.G, self.linkleg.L_r)
            self.set_rim(self.lower_rim_l, self.linkleg.F_l, self.linkleg.L_l)
            # five joints on the rims   (joint, center)
            self.set_joint(self.upper_joint_r, self.linkleg.H_r)
            self.set_joint(self.upper_joint_l, self.linkleg.H_l)
            self.set_joint(self.lower_joint_r, self.linkleg.F_r)
            self.set_joint(self.lower_joint_l, self.linkleg.F_l)
            self.set_joint(self.G_joint, self.linkleg.G)
            # six bars  (bar, point1, point2)
            self.set_bar(self.OB_bar_r, 0, self.linkleg.B_r)
            self.set_bar(self.OB_bar_l, 0, self.linkleg.B_l)
            self.set_bar(self.AE_bar_r, self.linkleg.A_r, self.linkleg.E)
            self.set_bar(self.AE_bar_l, self.linkleg.A_l, self.linkleg.E)
            self.set_bar(self.CD_bar_r, self.linkleg.C_r, self.linkleg.D_r)
            self.set_bar(self.CD_bar_l, self.linkleg.C_l, self.linkleg.D_l)
            
        def set_rim(self, rim, p1, o):  # rim, start point, center
            start = np.angle(p1-o, deg=True) 
            for arc in rim.arc: # inner & outer arcs
                arc.set_center([o.real, o.imag] + self.O)    # center(x, y)
                arc.set_angle( start - rim.start )    # rotate angle (degree)
                
        def set_joint(self, joint, center): # joint, center
            joint.set_center([center.real, center.imag] + self.O)
            
        def set_bar(self, bar, p1, p2): # bar, point1, point2
            bar.set_data([p1.real, p2.real] + self.O[0], [p1.imag, p2.imag] + self.O[1])

        
    ## Parameters Setting ##
    def setting(self, fig_size=-1, mark_size=-1, line_width=-1):
        if fig_size != -1:
            self.leg_shape.fig_size = fig_size
        if mark_size != -1:
            self.leg_shape.mark_size = mark_size
        if line_width != -1:
            self.leg_shape.line_width = line_width
            
    #### Plot leg with current shape ####
    def plot_leg(self, theta, beta, O, ax):
        self.O = np.array(O) # origin of leg in world coordinate
        self.leg_shape.O = np.array(O) # origin of leg in world coordinate
        leg_shape = self.leg_shape
        # initialize all graphics 
        self.forward(theta, beta, vector=False)  # update to apply displacement of origin of leg.
        leg_shape.get_shape()
        leg_shape.update()  # update to apply displacement of origin of leg.
        self.center_line, = ax.plot([], [], linestyle='--', color='blue', linewidth=1)   # center line
        self.joint_points = [ ax.plot([], [], 'ko', markersize=leg_shape.mark_size)[0] for _ in range(5) ]   # five dots at the center of joints
        # add leg part to the plot
        for key, value in leg_shape.__dict__.items():
            if "rim" in key:
                ax.add_patch(value.arc[0])
                ax.add_patch(value.arc[1])
            elif "joint" in key:
                ax.add_patch(value)
            elif "bar" in key:
                ax.add_line(value)
        # joint points
        for i, circle in enumerate([leg_shape.upper_joint_r, leg_shape.upper_joint_l, leg_shape.lower_joint_r, leg_shape.lower_joint_l, leg_shape.G_joint]):
            center = circle.get_center()
            self.joint_points[i].set_data([center[0]], [center[1]])
            
        return ax  
    
    #### Plot leg on one fig given from user ####
    def plot_one(self, theta=np.deg2rad(17.0), beta=0, O=np.array([0, 0]), ax=None): 
        O = np.array(O)
        if ax is None:
            fig, ax = plt.subplots()
        # plot setting
        ax.set_aspect('equal')  # 座標比例相同
        ax = self.plot_leg(theta, beta, O, ax)
        return ax

    #### Plot leg by given foothold of G, lower rim, or upper rim ####
    def plot_one_by_rim(self, O=np.array([0, 0]), foothold=np.array([0, 0]), rim='G', ax=None): 
        O = np.array(O)
        foothold = np.array(foothold)
        if rim == 'G':
            leg_length = np.linalg.norm(foothold - O + np.array([0, self.r]))
            theta = np.polyval(inv_G_dist_coef, leg_length)
            beta = np.angle( (foothold[0]-O[0] + 1j*(foothold[1]-O[1])) / -1j)
        elif rim == 'lower':
            O2 = foothold - O + np.array([0, self.radius])
            theta = np.polyval(inv_L_dist_coef, np.linalg.norm(O2))
            O2_x_beta0 = np.polyval(L_x_coef, theta)
            O2_y_beta0 = np.polyval(L_y_coef, theta)
            beta = np.angle( (O2[0] + 1j*O2[1]) / (O2_x_beta0 + 1j*O2_y_beta0))
        elif rim == 'upper':
            print("123")
            O1 = foothold - O + np.array([0, self.radius])
            theta = np.polyval(inv_U_dist_coef, np.linalg.norm(O1))
            O1_x_beta0 = np.polyval(U_x_coef, theta)
            O1_y_beta0 = np.polyval(U_y_coef, theta)
            beta = np.angle( (O1[0] + 1j*O1[1]) / (O1_x_beta0 + 1j*O1_y_beta0))
            
        if ax is None:
            fig, ax = plt.subplots()
        # plot setting
        ax.set_aspect('equal')  # 座標比例相同
        ax = self.plot_leg(theta, beta, O, ax)
        return ax

if __name__ == '__main__':
    file_name = 'plot_leg_example'
    
    LegAnimation = LegAnimation()  # rad
    ax = LegAnimation.plot_one()
    ax = LegAnimation.plot_one(np.deg2rad(130), np.deg2rad(45), [0., 0.3], ax=ax)
    ax = LegAnimation.plot_one_by_rim([0.2, 0.3], [0.3, 0.0], rim='G', ax=ax)
    ax = LegAnimation.plot_one_by_rim([0.5, 0.2], [0.6, 0.0], rim='lower', ax=ax)
    LegAnimation.setting(mark_size=10, line_width=3)
    ax = LegAnimation.plot_one_by_rim([0.7, 0.1], [0.8, 0.0], rim='upper', ax=ax)
    ax.grid()
    plt.savefig(file_name + '.png')
    plt.show()
    plt.close()

    