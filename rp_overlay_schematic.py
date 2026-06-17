import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from matplotlib import transforms

from PlotLeg import PlotLeg


def add_smooth_rim(ax, rim, color, alpha, zorder):
    center = np.asarray(rim.arc_fill.center)
    outer_radius = rim.arc_fill.r
    rim_width = rim.arc_fill.width
    inner_radius = outer_radius - rim_width
    mid_radius = 0.5 * (outer_radius + inner_radius)
    cap_radius = 0.5 * rim_width
    theta1 = np.deg2rad(rim.arc_fill.theta1)
    theta2 = np.deg2rad(rim.arc_fill.theta2)
    if theta2 < theta1:
        theta2 += 2 * np.pi

    theta = np.linspace(theta1, theta2, 160)
    outer = center + outer_radius * np.column_stack((np.cos(theta), np.sin(theta)))
    inner = center + inner_radius * np.column_stack((np.cos(theta[::-1]), np.sin(theta[::-1])))
    cap_end_center = center + mid_radius * np.array([np.cos(theta2), np.sin(theta2)])
    cap_start_center = center + mid_radius * np.array([np.cos(theta1), np.sin(theta1)])
    cap_end_theta = np.linspace(theta2, theta2 + np.pi, 40)
    cap_start_theta = np.linspace(theta1 + np.pi, theta1 + 2 * np.pi, 40)
    cap_end = cap_end_center + cap_radius * np.column_stack((np.cos(cap_end_theta), np.sin(cap_end_theta)))
    cap_start = cap_start_center + cap_radius * np.column_stack((np.cos(cap_start_theta), np.sin(cap_start_theta)))

    vertices = np.vstack((outer, cap_end, inner, cap_start, outer[:1]))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
    patch = PathPatch(
        Path(vertices, codes),
        facecolor="none",
        edgecolor=color,
        linewidth=3.0,
        alpha=alpha,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)


def plot_leg_rp_overlay():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 12,
        "mathtext.fontset": "stix",
    })

    file_name = "leg_rp_overlay"
    plot_leg = PlotLeg()
    plot_leg.setting(mark_size=5, line_width=1.5)

    theta = np.deg2rad(60)
    beta = np.deg2rad(-20)
    O = np.array([0.0, 0.0])
    plot_leg.forward(theta, beta, vector=False)
    plot_leg.leg_shape.get_shape(O)

    fig, ax = plt.subplots(figsize=(8, 8))

    arc_colors = {
        "upper_rim_r": "red",
        "upper_rim_l": "blue",
        "lower_rim_r": "green",
        "lower_rim_l": "purple",
        "joints": "black",
        "bars": "black",
    }

    leg_alpha = 0.24
    for key, value in plot_leg.leg_shape.__dict__.items():
        if "rim" in key:
            add_smooth_rim(ax, value, arc_colors.get(key, "black"), leg_alpha, 4)
        elif "joint" in key:
            continue
        elif "bar" in key:
            value.set_color("0.35")
            value.set_alpha(0.18)
            value.set_linewidth(1.2)
            value.set_marker("")
            value.set_zorder(3)
            ax.add_line(value)

    # RP equivalent model: R at O, P extension along OG.
    G = np.array([plot_leg.G.real, plot_leg.G.imag])
    og = G - O
    length = np.linalg.norm(og)
    direction = og / length
    normal = np.array([-direction[1], direction[0]])
    angle = np.rad2deg(np.arctan2(direction[1], direction[0]))

    # Revolute joint.
    ax.add_patch(Circle(O, 0.030, facecolor="white", edgecolor="black",
                        linewidth=2.3, zorder=20))
    ax.add_patch(Circle(O, 0.007, facecolor="black", edgecolor="black", zorder=21))

    # Prismatic body as a telescopic sleeve and sliding rod.
    sleeve_start = O + 0.020 * direction
    sleeve_length = 0.62 * length
    rod_start = O + 0.39 * length * direction
    rod_length = 0.58 * length

    sleeve_width = 0.026
    rod_width = 0.014
    rod = Rectangle((rod_start[0], rod_start[1] - rod_width / 2),
                    rod_length, rod_width,
                    facecolor="0.82", edgecolor="black", linewidth=1.8,
                    zorder=18)
    rod.set_transform(
        transforms.Affine2D()
        .rotate_deg_around(rod_start[0], rod_start[1], angle)
        + ax.transData
    )
    ax.add_patch(rod)

    sleeve = Rectangle((sleeve_start[0], sleeve_start[1] - sleeve_width / 2),
                       sleeve_length, sleeve_width,
                       facecolor=(1, 1, 1, 0.58), edgecolor="black", linewidth=2.4,
                       zorder=19)
    sleeve.set_transform(
        transforms.Affine2D()
        .rotate_deg_around(sleeve_start[0], sleeve_start[1], angle)
        + ax.transData
    )
    ax.add_patch(sleeve)

    ax.add_patch(Circle(G, 0.014, facecolor="white", edgecolor="black",
                        linewidth=1.9, zorder=22))
    ax.add_patch(Circle(G, 0.0048, facecolor="black", edgecolor="black", zorder=23))

    # Coordinate axes and labels.
    arrow_length = 0.1
    arrow_width = 0.01
    axis_alpha = 0.35
    ax.arrow(0, 0, arrow_length, 0, head_width=arrow_width, head_length=arrow_width * 1.5,
             fc="black", ec="black", alpha=axis_alpha, linewidth=1.0, zorder=1)
    ax.arrow(0, 0, 0, arrow_length, head_width=arrow_width, head_length=arrow_width * 1.5,
             fc="black", ec="black", alpha=axis_alpha, linewidth=1.0, zorder=1)

    ax.set_aspect("equal")
    ax.autoscale_view()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    axis_center = np.array([(xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2])
    axis_half_span = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    ax.set_xlim(axis_center[0] - axis_half_span, axis_center[0] + axis_half_span)
    ax.set_ylim(axis_center[1] - axis_half_span, axis_center[1] + axis_half_span)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.5, -0.08, "X(m)", transform=ax.transAxes, fontsize=20,
            fontfamily="Times New Roman", fontweight="bold", ha="center", va="top",
            clip_on=False, zorder=30)
    ax.text(-0.08, 0.5, "Z(m)", transform=ax.transAxes, fontsize=20,
            fontfamily="Times New Roman", fontweight="bold", ha="right", va="center",
            rotation=90, clip_on=False, zorder=30)

    plt.savefig(file_name + ".png", dpi=1000, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    plot_leg_rp_overlay()
