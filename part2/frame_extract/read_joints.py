import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

folder_name = "walking"

folder_path = os.path.join("outputs", "joints_npz", folder_name)

npz_files = [f for f in os.listdir(folder_path) if f.endswith(".npz")]
npz_files = sorted(npz_files, key=lambda f: int(os.path.splitext(f)[0]))

all_points = []
joint_names = None
times_ms = []

wanted_joints = [
    "left-acromion",
    "right-acromion",
    "left-hip",
    "right-hip",
    "left-knee",
    "right-knee",
    "left-ankle",
    "right-ankle",
    "left-heel",
    "right-heel",
    "left-small-toe-tip",
    "right-small-toe-tip",
    "left-big-toe-tip",
    "right-big-toe-tip",
]

keep_indices = None

for file_name in npz_files:
    file_path = os.path.join(folder_path, file_name)
    data = np.load(file_path, allow_pickle=True)

    points = data["person_0"]
    names = data["joint_names"]

    if points.ndim == 3:
        points = points[0]

    if keep_indices is None:
        keep_indices = [i for i, n in enumerate(names) if n in wanted_joints]
        joint_names = names[keep_indices]

    points = points[keep_indices]

    all_points.append(points)
    times_ms.append(int(os.path.splitext(file_name)[0]))

    if joint_names is None:
        joint_names = data["joint_names"]

name_to_idx = {name: i for i, name in enumerate(joint_names)}

connections = [
    ("left-acromion", "left-hip"),
    ("left-hip", "left-knee"),
    ("left-knee", "left-ankle"),
    ("left-ankle", "left-heel"),
    ("left-heel", "left-small-toe-tip"),
    ("left-heel", "left-big-toe-tip"),
    ("left-small-toe-tip", "left-big-toe-tip"),

    ("right-acromion", "right-hip"),
    ("right-hip", "right-knee"),
    ("right-knee", "right-ankle"),
    ("right-ankle", "right-heel"),
    ("right-heel", "right-small-toe-tip"),
    ("right-heel", "right-big-toe-tip"),
    ("right-small-toe-tip", "right-big-toe-tip"),

    ("left-acromion", "right-acromion"),
    ("left-hip", "right-hip"),
]

all_points = np.array(all_points)
times_ms = np.array(times_ms)

x_all = all_points[:, :, 0]
y_all = all_points[:, :, 1]
z_all = all_points[:, :, 2]

max_range = np.array([
    x_all.max() - x_all.min(),
    y_all.max() + 0.2 - y_all.min(),
    z_all.max() - z_all.min()
]).max() / 2

mid_x = (x_all.max() + x_all.min()) * 0.5
mid_y = (y_all.max() + y_all.min()) * 0.5
mid_z = (z_all.max() + z_all.min()) * 0.5

fig = plt.figure(figsize=(8, 7))
ax = fig.add_subplot(111, projection="3d")

fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.set_xlim(mid_x - max_range, mid_x + max_range)
ax.set_ylim(mid_y - max_range, mid_y + max_range)
ax.set_zlim(mid_z - max_range, mid_z + max_range)
ax.set_box_aspect([1, 1, 1])

ax.view_init(elev=-49, azim=-7, roll=-84)
ax.grid(True, alpha=0.3)
ax.set_xlabel("X", fontsize=20)
ax.set_ylabel("Y", fontsize=20)
ax.set_zlabel("Z", fontsize=20)

points0 = all_points[0]

scatter = ax.scatter(
    points0[:, 0],
    points0[:, 1],
    points0[:, 2],
    s=80,
    color="#2ca9df",
    edgecolors="black",
    linewidth=0.5
)

line_objects = []
for joint_a, joint_b in connections:
    if joint_a in name_to_idx and joint_b in name_to_idx:
        ia = name_to_idx[joint_a]
        ib = name_to_idx[joint_b]

        line, = ax.plot(
            [points0[ia, 0], points0[ib, 0]],
            [points0[ia, 1], points0[ib, 1]],
            [points0[ia, 2], points0[ib, 2]],
            color="black",
            linewidth=2
        )
        line_objects.append((line, ia, ib))

bar_ax = fig.add_axes([0.08, 0.04, 0.84, 0.025])
bar_ax.set_xlim(0, 1)
bar_ax.set_ylim(0, 1)
bar_ax.axis("off")

bar_bg = Rectangle((0, 0.2), 1, 0.6, facecolor="#e6e6e6", edgecolor="none")
bar_fg = Rectangle((0, 0.2), 0, 0.6, facecolor="#2ca9df", edgecolor="none")

bar_ax.add_patch(bar_bg)
bar_ax.add_patch(bar_fg)

total_time_ms = times_ms[-1] if times_ms[-1] > 0 else 1
total_time_s = total_time_ms / 1000

time_text = fig.text(
    0.92, 0.1,
    f"{times_ms[0] / 1000:.2f}s / {total_time_s:.2f}s",
    ha="right",
    va="center",
    fontsize=32,
    color="#444444"
)

def update(frame):
    points = all_points[frame]

    scatter._offsets3d = (points[:, 0], points[:, 1], points[:, 2])

    for line, ia, ib in line_objects:
        line.set_data(
            [points[ia, 0], points[ib, 0]],
            [points[ia, 1], points[ib, 1]]
        )
        line.set_3d_properties([points[ia, 2], points[ib, 2]])

    current_time_ms = times_ms[frame]
    current_time_s = current_time_ms / 1000
    progress = current_time_ms / total_time_ms
    bar_fg.set_width(progress)

    time_text.set_text(f"{current_time_s:.2f}s / {total_time_s:.2f}s")

    return [scatter] + [line for line, _, _ in line_objects] + [bar_fg, time_text]

ani = FuncAnimation(
    fig,
    update,
    frames=len(all_points),
    interval=62.5,
    blit=False,
    repeat=True
)

ani.save("skeleton_animation.mp4", fps=16, dpi=300)

plt.tight_layout(rect=[0, 0.08, 1, 1])
plt.show()