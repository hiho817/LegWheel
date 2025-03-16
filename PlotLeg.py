import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
from matplotlib.lines import Line2D
import time
import LegModel
from FittedCoefficient import *

class PlotLeg(LegModel.LegModel):
    def __init__(self, sim=True):
        super().__init__(sim=sim)
        self.forward(np.deg2rad(17), 0, vector=False)
        self.O = np.array([0, 0])   # origin of leg in world coordinate
        self.leg_shape = self.LegShape(self, self.O)   # initial pose of leg
        
    ## Shape Of Leg ##  
    class LegShape:
        def __init__(self, leg_model, O):
            self.O = O
            self.leg_model = leg_model
            self.R = leg_model.R
            self.r = leg_model.r
            self.radius = leg_model.radius
            # Plot setting 
            self.fig_size = 10
            self.mark_size = 2.0
            self.line_width = 1.0
            self.zorder = 1.0
            self.color = "black"
            self.get_shape()
            
        class rim:
            def __init__(self, arc, arc_out, start):
                self.arc = [arc, arc_out]   # inner & outer arcs
                self.start = start          # start angle 
                
        def get_shape(self):
            # four rims (inner arc, outer arc, start angle)
            self.upper_rim_r = self.rim( *self.get_arc(self.leg_model.F_r, self.leg_model.H_r, self.leg_model.U_r, self.r))
            self.upper_rim_l = self.rim( *self.get_arc(self.leg_model.H_l, self.leg_model.F_l, self.leg_model.U_l, self.r))
            self.lower_rim_r = self.rim( *self.get_arc(self.leg_model.G,   self.leg_model.F_r, self.leg_model.L_r, self.r))
            self.lower_rim_l = self.rim( *self.get_arc(self.leg_model.F_l, self.leg_model.G,   self.leg_model.L_l, self.r))
            # five joints on the rims   (center, radius)
            self.upper_joint_r = self.get_circle(self.leg_model.H_r, self.r) 
            self.upper_joint_l = self.get_circle(self.leg_model.H_l, self.r) 
            self.lower_joint_r = self.get_circle(self.leg_model.F_r, self.r) 
            self.lower_joint_l = self.get_circle(self.leg_model.F_l, self.r) 
            self.G_joint       = self.get_circle(self.leg_model.G,   self.r)
            # six bars  (point1, point2)
            self.OB_bar_r = self.get_line(0, self.leg_model.B_r) 
            self.OB_bar_l = self.get_line(0, self.leg_model.B_l) 
            self.AE_bar_r = self.get_line(self.leg_model.A_r, self.leg_model.E)
            self.AE_bar_l = self.get_line(self.leg_model.A_l, self.leg_model.E)
            self.CD_bar_r = self.get_line(self.leg_model.C_r, self.leg_model.D_r)
            self.CD_bar_l = self.get_line(self.leg_model.C_l, self.leg_model.D_l) 
            
        def get_arc(self, p1, p2, o, offset=0.01):
            start = np.angle(p1-o, deg=True)
            end = np.angle(p2-o, deg=True)
            radius = np.abs(p1-o)
            arc = Arc([o.real, o.imag], 2*(radius-offset), 2*(radius-offset), angle=0.0, theta1=start, theta2=end, color=self.color, linewidth=self.line_width, zorder=self.zorder)
            arc_out = Arc([o.real, o.imag], 2*(radius+offset), 2*(radius+offset), angle=0.0, theta1=start, theta2=end, color=self.color, linewidth=self.line_width, zorder=self.zorder)
            return arc, arc_out, start

        def get_circle(self, o, r):
            circle = Arc([o.real, o.imag], 2*r, 2*r, angle=0.0, theta1=0, theta2=360, color=self.color, linewidth=self.line_width, zorder=self.zorder)
            return circle

        def get_line(self, p1, p2):
            line = Line2D([p1.real, p2.real], [p1.imag, p2.imag], marker='o', markersize=self.mark_size, linestyle='-', color=self.color, linewidth=self.line_width, zorder=self.zorder)
            return line
        
        ## Set Postion Of Leg ##  
        def set_shape(self, O):
            self.O = np.array(O)    # origin of leg in world coordinate
            # four rims (rim, start point, center)
            self.set_rim(self.upper_rim_r, self.leg_model.F_r, self.leg_model.U_r)
            self.set_rim(self.upper_rim_l, self.leg_model.H_l, self.leg_model.U_l)
            self.set_rim(self.lower_rim_r, self.leg_model.G, self.leg_model.L_r)
            self.set_rim(self.lower_rim_l, self.leg_model.F_l, self.leg_model.L_l)
            # five joints on the rims   (joint, center)
            self.set_joint(self.upper_joint_r, self.leg_model.H_r)
            self.set_joint(self.upper_joint_l, self.leg_model.H_l)
            self.set_joint(self.lower_joint_r, self.leg_model.F_r)
            self.set_joint(self.lower_joint_l, self.leg_model.F_l)
            self.set_joint(self.G_joint, self.leg_model.G)
            # six bars  (bar, point1, point2)
            self.set_bar(self.OB_bar_r, 0, self.leg_model.B_r)
            self.set_bar(self.OB_bar_l, 0, self.leg_model.B_l)
            self.set_bar(self.AE_bar_r, self.leg_model.A_r, self.leg_model.E)
            self.set_bar(self.AE_bar_l, self.leg_model.A_l, self.leg_model.E)
            self.set_bar(self.CD_bar_r, self.leg_model.C_r, self.leg_model.D_r)
            self.set_bar(self.CD_bar_l, self.leg_model.C_l, self.leg_model.D_l)
            
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
    def setting(self, fig_size=-1, mark_size=-1, line_width=-1, color=None, zorder=None):
        if fig_size != -1:
            self.leg_shape.fig_size = fig_size
        if mark_size != -1:
            self.leg_shape.mark_size = mark_size
        if line_width != -1:
            self.leg_shape.line_width = line_width
        if color != None:
            self.leg_shape.color = color
        if zorder != None:
            self.leg_shape.zorder = zorder
            
    #### Plot leg with current shape ####
    def plot_leg(self, theta, beta, O, ax):
        # initialize all graphics 
        self.forward(theta, beta, vector=False)  # update to apply displacement of origin of leg.
        self.leg_shape.get_shape()
        self.leg_shape.set_shape(O)  # set to apply displacement of origin of leg.
        # self.center_line, = ax.plot([], [], linestyle='--', color='blue', linewidth=1)   # center line (Axis of Symmetry)
        self.joint_points = [ ax.plot([], [], marker='o', color=self.leg_shape.color, markersize=self.leg_shape.mark_size, zorder=self.leg_shape.zorder)[0] for _ in range(5) ]   # five dots at the center of joints
        # add leg part to the plot
        for key, value in self.leg_shape.__dict__.items():
            if "rim" in key:
                ax.add_patch(value.arc[0])
                ax.add_patch(value.arc[1])
            elif "joint" in key:
                ax.add_patch(value)
            elif "bar" in key:
                ax.add_line(value)
        # joint points
        for i, circle in enumerate([self.leg_shape.upper_joint_r, self.leg_shape.upper_joint_l, self.leg_shape.lower_joint_r, self.leg_shape.lower_joint_l, self.leg_shape.G_joint]):
            center = circle.get_center()
            self.joint_points[i].set_data([center[0]], [center[1]])
            
        return ax  
    
    #### Plot leg on one fig given from user ####
    def plot_by_angle(self, theta=np.deg2rad(17.0), beta=0, O=np.array([0, 0]), ax=None): 
        O = np.array(O)
        if ax is None:
            fig, ax = plt.subplots()
        # plot setting
        ax.set_aspect('equal')  # 座標比例相同
        ax = self.plot_leg(theta, beta, O, ax)
        return ax

    #### Plot leg by given foothold of G, lower rim, or upper rim ####
    def plot_by_rim(self, foothold=np.array([0, 0]), O=np.array([0, 0]), rim='G', ax=None): 
        O = np.array(O)
        foothold = np.array(foothold)
        if rim == 'G':
            theta, beta = self.inverse(foothold - O + np.array([0, self.r]), 'G')
        elif rim == 'lower':
            if foothold[0] > O[0]:  # left lower rim
                theta, beta = self.inverse(foothold - O + np.array([0, self.radius]), 'L_l')
            else:                   # right lower rim
                theta, beta = self.inverse(foothold - O + np.array([0, self.radius]), 'L_r')
        elif rim == 'upper':
            if foothold[0] > O[0]:  # left lower rim
                theta, beta = self.inverse(foothold - O + np.array([0, self.radius]), 'U_l')
            else:                   # right lower rim
                theta, beta = self.inverse(foothold - O + np.array([0, self.radius]), 'U_r')
            
        if ax is None:
            fig, ax = plt.subplots()
        # plot setting
        ax.set_aspect('equal')  # 座標比例相同
        ax = self.plot_leg(theta, beta, O, ax)
        return ax
    

if __name__ == '__main__':
    file_name = 'plot_leg_example'
    start_time = time.time()  # end time
    
    plot_leg = PlotLeg()  # rad
    ax = plot_leg.plot_by_angle()
    ax = plot_leg.plot_by_angle(np.deg2rad(130), np.deg2rad(-45), [0., 0.3], ax=ax)
    ax = plot_leg.plot_by_rim([0.2, 0.0], [0.1, 0.3], rim='G', ax=ax)
    ax = plot_leg.plot_by_rim([0.6, 0.1], [0.5, 0.2], rim='lower', ax=ax)
    ax = plot_leg.plot_by_rim([0.3, 0.1], [0.4, 0.2], rim='lower', ax=ax)
    plot_leg.setting(mark_size=10, line_width=3, color='red')
    ax = plot_leg.plot_by_rim([0.8, 0.0], [0.9, 0.12], rim='upper', ax=ax)
    ax = plot_leg.plot_by_rim([1.0, 0.0], [0.9, 0.12], rim='upper', ax=ax)
    ax.grid()
    
    plt.savefig(file_name + '.png')
    
    end_time = time.time()  # end time
    print("\nExecution Time:", end_time - start_time, "seconds")
    
    plt.show()
    plt.close()

    
