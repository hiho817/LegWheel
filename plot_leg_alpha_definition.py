"""Plot the leg rim parameterization used in the thesis.

The main-rim coordinate alpha increases from the left upper rim
(alpha = -pi) to the right upper rim (alpha = pi).  The exposed toe
surface around G uses a separate local coordinate alpha_G in [0, pi].
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Arc, Circle, FancyArrowPatch
from matplotlib.path import Path as MplPath

from PlotLeg import PlotLeg


THETA_DEG = 17.0
BETA_DEG = 0.0
SCRIPT_DIR = (
    Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd().resolve()
)
OUTPUT_FILE = SCRIPT_DIR / "leg_alpha_definition.pdf"
LINE_EFFECTS = [
    path_effects.Stroke(linewidth=3.6, foreground="white"),
    path_effects.Normal(),
]


def complex_xy(point):
    return np.array([point.real, point.imag], dtype=float)


def sample_rim_centerline(rim, samples=180):
    """Return points along the middle of a PlotLeg annular rim."""
    wedge = rim.arc_fill
    center = np.asarray(wedge.center, dtype=float)
    radius = wedge.r - 0.5 * wedge.width
    angle_1 = np.deg2rad(wedge.theta1)
    angle_2 = np.deg2rad(wedge.theta2)
    if angle_2 < angle_1:
        angle_2 += 2.0 * np.pi
    angles = np.linspace(angle_1, angle_2, samples)
    return center + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def add_alpha_rim(
    ax,
    rim,
    alpha_start,
    alpha_end,
    cmap,
    norm,
    half_width,
    trim_start=0,
    trim_end=0,
):
    """Draw one rim with color encoding the global alpha coordinate."""
    points = sample_rim_centerline(rim)
    if trim_end:
        points = points[trim_start:-trim_end]
    else:
        points = points[trim_start:]

    center = np.asarray(rim.arc_fill.center, dtype=float)
    radial = points - center
    radial /= np.linalg.norm(radial, axis=1, keepdims=True)
    outer = points + half_width * radial
    inner = points - half_width * radial
    polygons = [
        [outer[i], outer[i + 1], inner[i + 1], inner[i]]
        for i in range(len(points) - 1)
    ]
    alpha_values = np.linspace(alpha_start, alpha_end, len(polygons))
    collection = PolyCollection(
        polygons,
        facecolors=cmap(norm(alpha_values)),
        edgecolors="none",
        zorder=9,
    )
    ax.add_collection(collection)
    ax.plot(outer[:, 0], outer[:, 1], color="0.25", linewidth=1.8, zorder=10)
    ax.plot(inner[:, 0], inner[:, 1], color="0.25", linewidth=1.8, zorder=10)
    return points


def add_straight_top_bridge(ax, left, right, half_width, cmap, norm):
    """Close the top seam with a straight band instead of overlapping arcs."""
    midpoint = 0.5 * (left + right)
    tangent = right - left
    tangent /= np.linalg.norm(tangent)
    normal = np.array([-tangent[1], tangent[0]])
    if normal[1] < 0:
        normal *= -1

    for start, end, alpha_value in (
        (left, midpoint, -np.pi),
        (midpoint, right, np.pi),
    ):
        polygon = [
            start + half_width * normal,
            end + half_width * normal,
            end - half_width * normal,
            start - half_width * normal,
        ]
        ax.add_collection(
            PolyCollection(
                [polygon],
                facecolors=[cmap(norm(alpha_value))],
                edgecolors="none",
                zorder=9,
            )
        )

    outer = np.vstack((left + half_width * normal, right + half_width * normal))
    inner = np.vstack((left - half_width * normal, right - half_width * normal))
    ax.plot(outer[:, 0], outer[:, 1], color="0.25", linewidth=1.8, zorder=10)
    ax.plot(inner[:, 0], inner[:, 1], color="0.25", linewidth=1.8, zorder=10)


def add_direction_arrow(ax, points, color="0.15"):
    """Add an arrow exactly along the rim centerline."""
    i0 = int(0.43 * (len(points) - 1))
    i1 = int(0.57 * (len(points) - 1))
    arrow_points = points[i0 : i1 + 1]
    arrow_path = MplPath(
        arrow_points,
        [MplPath.MOVETO] + [MplPath.LINETO] * (len(arrow_points) - 1),
    )
    arrow = FancyArrowPatch(
        path=arrow_path,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color=color,
        zorder=12,
    )
    arrow.set_path_effects(LINE_EFFECTS)
    ax.add_patch(arrow)


def circular_arc_points(center, radius, start_angle, delta_angle, samples=100):
    angles = np.linspace(start_angle, start_angle + delta_angle, samples)
    return center + radius * np.column_stack((np.cos(angles), np.sin(angles)))


def add_gradient_curve(ax, points, values, cmap, norm, linewidth, zorder):
    segments = np.stack((points[:-1], points[1:]), axis=1)
    collection = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        capstyle="round",
        zorder=zorder,
    )
    collection.set_array(np.asarray(values))
    ax.add_collection(collection)
    return collection


def plot_alpha_definition():
    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 12,
            "mathtext.fontset": "stix",
        }
    )

    leg = PlotLeg()
    leg.setting(mark_size=5, line_width=1.5)
    leg.forward(np.deg2rad(THETA_DEG), np.deg2rad(BETA_DEG), vector=False)
    leg.leg_shape.get_shape(np.array([0.0, 0.0]))

    fig, ax = plt.subplots(figsize=(9.2, 8.0))
    cmap = plt.get_cmap("coolwarm")
    norm = Normalize(vmin=-np.pi, vmax=np.pi)

    # Bars are kept neutral so the alpha encoding remains visually dominant.
    for key, value in leg.leg_shape.__dict__.items():
        if "bar" in key:
            value.set_color("0.25")
            value.set_linewidth(1.5)
            value.set_zorder(3)
            ax.add_line(value)

    # The breakpoints follow the model's 130 deg upper and 50 deg lower rims.
    alpha_f = np.deg2rad(50.0)
    rim_specs = [
        (leg.leg_shape.upper_rim_l, -np.pi, -alpha_f, 0, 0),
        (leg.leg_shape.lower_rim_l, -alpha_f, 0.0, 0, 0),
        (leg.leg_shape.lower_rim_r, 0.0, alpha_f, 0, 0),
        (leg.leg_shape.upper_rim_r, alpha_f, np.pi, 0, 0),
    ]
    rim_paths = []
    for rim, alpha_start, alpha_end, trim_start, trim_end in rim_specs:
        path = add_alpha_rim(
            ax,
            rim,
            alpha_start,
            alpha_end,
            cmap,
            norm,
            half_width=leg.r,
            trim_start=trim_start,
            trim_end=trim_end,
        )
        add_direction_arrow(ax, path)
        rim_paths.append(path)

    # Close the -pi / pi seam with a black radial edge.
    top_seam = 0.5 * (rim_paths[0][0] + rim_paths[3][-1])
    seam_line, = ax.plot(
        [top_seam[0], top_seam[0]],
        [top_seam[1] - leg.r, top_seam[1] + leg.r],
        color="0.15",
        linewidth=1.8,
        zorder=11,
    )

    # Joint outlines and centers.
    joint_points = [leg.F_l, leg.G, leg.F_r]
    for point in joint_points:
        xy = complex_xy(point)
        ax.add_patch(
            Circle(
                xy,
                radius=leg.r,
                facecolor="white",
                edgecolor="0.15",
                linewidth=1.8,
                zorder=11,
            )
        )
        ax.plot(*xy, "o", color="black", markersize=4.5, zorder=13)

    g = complex_xy(leg.G)

    # The local coordinate follows the right semicircle around G:
    # alpha_G = 0 at the bottom and alpha_G = pi at the top.
    local_radius = leg.r
    local_arc = circular_arc_points(
        g, local_radius, -0.5 * np.pi, np.pi, samples=180
    )
    local_cmap = plt.get_cmap("YlOrBr")
    local_norm = Normalize(vmin=0.0, vmax=np.pi)
    ax.plot(
        local_arc[:, 0],
        local_arc[:, 1],
        color="0.25",
        linewidth=3.2,
        solid_capstyle="round",
        zorder=13,
    )
    add_gradient_curve(
        ax,
        local_arc,
        np.linspace(0.0, np.pi, len(local_arc) - 1),
        local_cmap,
        local_norm,
        linewidth=1.7,
        zorder=14,
    )
    local_arrow_radius = 1.48 * leg.r
    local_arrow_arc = circular_arc_points(
        g, local_arrow_radius, -0.5 * np.pi, np.pi, samples=180
    )
    local_arrow_points = local_arrow_arc[72:107]
    local_arrow_path = MplPath(
        local_arrow_points,
        [MplPath.MOVETO]
        + [MplPath.LINETO] * (len(local_arrow_points) - 1),
    )
    local_arrow = FancyArrowPatch(
        path=local_arrow_path,
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.3,
        color="0.15",
        zorder=15,
    )
    local_arrow.set_path_effects(LINE_EFFECTS)
    ax.add_patch(local_arrow)
    ax.text(
        g[0] + 0.002,
        g[1] - leg.r - 0.004,
        r"$0$",
        color="black",
        fontweight="bold",
        ha="left",
        va="top",
        zorder=16,
    )
    ax.text(
        g[0] + 0.002,
        g[1] + leg.r + 0.004,
        r"$\pi$",
        color="black",
        fontweight="bold",
        ha="left",
        va="bottom",
        zorder=16,
    )
    g_annotation = ax.annotate(
        r"$G$",
        xy=(g[0], g[1]),
        xytext=(g[0] - 0.014, g[1] - leg.r - 0.014),
        ha="center",
        va="top",
        arrowprops=dict(
            arrowstyle="-",
            color="black",
            linewidth=1.2,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=16,
    )
    g_annotation.arrow_patch.set_path_effects(LINE_EFFECTS)

    # Endpoint labels make the alpha convention explicit without crowding
    # every intermediate point.
    label_specs = [
        (rim_paths[0][0], r"$\alpha=-\pi$", (-0.012, 0.020), "right"),
        (rim_paths[0][-1], r"$-\frac{5\pi}{18}$", (-0.022, -0.018), "right"),
        (rim_paths[3][0], r"$\frac{5\pi}{18}$", (0.022, -0.018), "left"),
        (rim_paths[3][-1], r"$\alpha=\pi$", (0.012, 0.020), "left"),
    ]
    for point, label, offset, horizontal_alignment in label_specs:
        annotation = ax.annotate(
            label,
            xy=point,
            xytext=point + np.asarray(offset),
            ha=horizontal_alignment,
            va="center",
            fontsize=13,
            fontweight="bold",
            arrowprops=dict(arrowstyle="-", color="0.35", lw=0.9),
            zorder=15,
        )
        annotation.arrow_patch.set_path_effects(LINE_EFFECTS)

    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label(r"Main-rim coordinate $\alpha$", fontsize=13)
    colorbar.set_ticks([-np.pi, -alpha_f, 0.0, alpha_f, np.pi])
    colorbar.set_ticklabels(
        [
            r"$-\pi$",
            r"$-\frac{5\pi}{18}$",
            r"$0$",
            r"$\frac{5\pi}{18}$",
            r"$\pi$",
        ]
    )

    ax.set_title(
        rf"Rim parameterization ($\theta={THETA_DEG:.0f}^\circ$, "
        rf"$\beta={BETA_DEG:.0f}^\circ$)",
        fontsize=15,
        pad=12,
    )
    ax.set_xlabel(r"$X$ (m)", fontsize=15)
    ax.set_ylabel(r"$Z$ (m)", fontsize=15)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.tick_params(
        axis="both",
        which="both",
        labelbottom=False,
        labelleft=False,
        bottom=False,
        left=False,
    )
    ax.margins(0.15)

    fig.tight_layout()
    fig.savefig(OUTPUT_FILE, bbox_inches="tight")
    return fig, ax


if __name__ == "__main__":
    figure, _ = plot_alpha_definition()
    plt.close(figure)
    print(f"Saved: {OUTPUT_FILE}")
