import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle
from matplotlib.patches import PathPatch
from matplotlib.path import Path

from PlotLeg import PlotLeg


OUTPUT_FILE = "運動學.svg"
RIM_COLOR = "#000000"
LOWER_RIGHT_COLOR = "#0059FF"
LINK_EDGE_COLOR = "#2B2F33"
LINK_SLOT_COLOR = "#D8DCDF"
STRUCTURE_COLOR = "#2B2F33"
JOINT_FILL_COLOR = "#A8B0B6"
TRANSPARENT_BACKGROUND = True
LINK_EDGE_WIDTH = 20
LINK_SLOT_WIDTH = 15
JOINT_RADIUS = 0.0025
RIM_JOINT_RADIUS_SCALE = 0.5
RIM_JOINT_CENTER_SCALE = 0.38
LOWER_RIGHT_CENTER_POINT_RADIUS = 0.0032


def add_smooth_rim(ax, rim, color, zorder):
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
    inner = center + inner_radius * np.column_stack(
        (np.cos(theta[::-1]), np.sin(theta[::-1]))
    )
    cap_end_center = center + mid_radius * np.array([np.cos(theta2), np.sin(theta2)])
    cap_start_center = center + mid_radius * np.array([np.cos(theta1), np.sin(theta1)])
    cap_end_theta = np.linspace(theta2, theta2 + np.pi, 40)
    cap_start_theta = np.linspace(theta1 + np.pi, theta1 + 2 * np.pi, 40)
    cap_end = cap_end_center + cap_radius * np.column_stack(
        (np.cos(cap_end_theta), np.sin(cap_end_theta))
    )
    cap_start = cap_start_center + cap_radius * np.column_stack(
        (np.cos(cap_start_theta), np.sin(cap_start_theta))
    )

    vertices = np.vstack((outer, cap_end, inner, cap_start, outer[:1]))
    codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
    patch = PathPatch(
        Path(vertices, codes),
        facecolor=color,
        edgecolor=color,
        linewidth=3.6,
        capstyle="round",
        joinstyle="round",
        zorder=zorder,
    )
    ax.add_patch(patch)
    outline = PathPatch(
        Path(vertices, codes),
        facecolor="none",
        edgecolor=color,
        linewidth=3.8,
        capstyle="round",
        joinstyle="round",
        zorder=zorder + 0.1,
    )
    ax.add_patch(outline)


def add_outer_dashed_rim_arc(ax, rim, color, zorder):
    center = np.asarray(rim.arc_fill.center, dtype=float)
    radius = rim.arc_fill.r
    theta1 = np.deg2rad(rim.arc_fill.theta1)
    theta2 = np.deg2rad(rim.arc_fill.theta2)
    if theta2 < theta1:
        theta2 += 2 * np.pi

    theta = np.linspace(theta1, theta2, 160)
    arc = center + radius * np.column_stack((np.cos(theta), np.sin(theta)))
    ax.plot(
        arc[:, 0],
        arc[:, 1],
        color=color,
        linestyle="--",
        linewidth=3.5,
        zorder=zorder,
    )


def add_slot_link(ax, line, zorder):
    x_data = np.asarray(line.get_xdata(), dtype=float)
    y_data = np.asarray(line.get_ydata(), dtype=float)
    ax.plot(
        x_data,
        y_data,
        color=LINK_EDGE_COLOR,
        linewidth=LINK_EDGE_WIDTH,
        solid_capstyle="round",
        zorder=zorder,
    )
    ax.plot(
        x_data,
        y_data,
        color=LINK_SLOT_COLOR,
        linewidth=LINK_SLOT_WIDTH,
        solid_capstyle="round",
        zorder=zorder + 0.1,
    )


def add_joint_axis(ax, center, zorder):
    ax.add_patch(
        Circle(
            center,
            JOINT_RADIUS,
            facecolor=JOINT_FILL_COLOR,
            edgecolor=STRUCTURE_COLOR,
            linewidth=1.2,
            zorder=zorder,
        )
    )


def add_rim_joint_axis(ax, center, radius, zorder):
    outer_radius = radius * RIM_JOINT_RADIUS_SCALE
    ax.add_patch(
        Circle(
            center,
            outer_radius,
            facecolor=JOINT_FILL_COLOR,
            edgecolor=STRUCTURE_COLOR,
            linewidth=1.8,
            zorder=zorder,
        )
    )
    ax.add_patch(
        Circle(
            center,
            outer_radius * RIM_JOINT_CENTER_SCALE,
            facecolor=STRUCTURE_COLOR,
            edgecolor=STRUCTURE_COLOR,
            linewidth=0.8,
            zorder=zorder + 0.1,
        )
    )


def add_lower_radius_lines(ax, rim, color, zorder):
    rim_center = np.asarray(rim.arc_fill.center, dtype=float)
    radius = rim.arc_fill.r
    theta1 = np.deg2rad(rim.arc_fill.theta1)
    theta2 = np.deg2rad(rim.arc_fill.theta2)
    for theta in (theta1, theta2):
        endpoint = rim_center + radius * np.array([np.cos(theta), np.sin(theta)])
        ax.plot(
            [rim_center[0], endpoint[0]],
            [rim_center[1], endpoint[1]],
            color=color,
            linestyle="--",
            linewidth=3.5,
            zorder=zorder,
        )


def add_center_point(ax, center, color, zorder):
    ax.add_patch(
        Circle(
            center,
            LOWER_RIGHT_CENTER_POINT_RADIUS,
            facecolor=color,
            edgecolor=color,
            linewidth=0,
            zorder=zorder,
        )
    )


def plot_leg_with_key_points(output_file=OUTPUT_FILE):
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "mathtext.fontset": "stix",
        }
    )

    plot_leg = PlotLeg()
    arc_colors = {
        "upper_rim_r": RIM_COLOR,
        "upper_rim_l": RIM_COLOR,
        "lower_rim_r": RIM_COLOR,
        "lower_rim_l": RIM_COLOR,
        "joints": STRUCTURE_COLOR,
        "bars": LINK_EDGE_COLOR,
    }

    plot_leg.setting(mark_size=5, line_width=1.5)
    fig, ax = plt.subplots(figsize=(8, 8))
    if TRANSPARENT_BACKGROUND:
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)

    theta = np.deg2rad(60)
    beta = np.deg2rad(-20)
    origin = np.array([0, 0])

    plot_leg.forward(theta, beta, vector=False)
    plot_leg.leg_shape.get_shape(origin)

    rim_zorder = 12
    point_zorder = 14

    for key, value in plot_leg.leg_shape.__dict__.items():
        if "rim" in key:
            rim_color = arc_colors.get(key, "black")
            add_smooth_rim(ax, value, rim_color, rim_zorder)
        elif "joint" in key:
            continue
        elif "bar" in key:
            add_slot_link(ax, value, zorder=7)

    rim_joint_circles = (
        plot_leg.leg_shape.upper_joint_r,
        plot_leg.leg_shape.upper_joint_l,
        plot_leg.leg_shape.lower_joint_r,
        plot_leg.leg_shape.lower_joint_l,
        plot_leg.leg_shape.G_joint,
    )

    joint_centers = []
    for circle in (
        *rim_joint_circles,
    ):
        joint_centers.append(circle.get_center())

    for value in plot_leg.leg_shape.__dict__.values():
        if hasattr(value, "get_xdata") and hasattr(value, "get_ydata"):
            x_data = np.asarray(value.get_xdata(), dtype=float)
            y_data = np.asarray(value.get_ydata(), dtype=float)
            if len(x_data) == 2 and len(y_data) == 2:
                joint_centers.extend(zip(x_data, y_data))

    unique_joint_centers = []
    for center in joint_centers:
        center_array = np.asarray(center, dtype=float)
        if not any(np.allclose(center_array, existing, atol=1e-9) for existing in unique_joint_centers):
            unique_joint_centers.append(center_array)

    for center in unique_joint_centers:
        add_joint_axis(ax, center, point_zorder)

    add_lower_radius_lines(
        ax,
        plot_leg.leg_shape.lower_rim_r,
        LOWER_RIGHT_COLOR,
        point_zorder + 1.0,
    )
    add_outer_dashed_rim_arc(
        ax,
        plot_leg.leg_shape.lower_rim_r,
        LOWER_RIGHT_COLOR,
        point_zorder + 1.0,
    )

    for circle in rim_joint_circles:
        add_rim_joint_axis(
            ax,
            circle.get_center(),
            circle.get_radius(),
            point_zorder + 0.3,
        )

    add_center_point(
        ax,
        plot_leg.leg_shape.lower_rim_r.arc_fill.center,
        LOWER_RIGHT_COLOR,
        point_zorder + 1.1,
    )

    ax.set_aspect("equal")
    ax.autoscale_view()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    axis_center = np.array([(xlim[0] + xlim[1]) / 2, (ylim[0] + ylim[1]) / 2])
    axis_half_span = 0.5 * max(xlim[1] - xlim[0], ylim[1] - ylim[0])
    ax.set_xlim(axis_center[0] - axis_half_span, axis_center[0] + axis_half_span)
    ax.set_ylim(axis_center[1] - axis_half_span, axis_center[1] + axis_half_span)
    ax.set_axis_off()

    fig.savefig(output_file, bbox_inches="tight", transparent=TRANSPARENT_BACKGROUND)
    return fig, ax


if __name__ == "__main__":
    figure, _ = plot_leg_with_key_points()
    plt.close(figure)
    print(f"Saved: {OUTPUT_FILE}")
