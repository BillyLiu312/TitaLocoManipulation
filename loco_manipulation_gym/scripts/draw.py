from loco_manipulation_gym.envs.tita.tita_config import TitaRoughCfg
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# 加载配置
cfg = TitaRoughCfg.goal_ee

# 球坐标 -> 笛卡尔坐标转换
def sphere2cart(r, theta, phi):
    x = r * np.cos(phi) * np.cos(theta)
    y = r * np.sin(phi)
    z = r * np.cos(phi) * np.sin(theta)
    return [x, y, z]

# 创建图形
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# 绘制立方体函数（添加label参数）
def plot_cube(ax, lower, upper, color='red', alpha=0.1, label=None):
    vertices = np.array([
        [lower[0], lower[1], lower[2]],  # 0
        [upper[0], lower[1], lower[2]],  # 1
        [upper[0], upper[1], lower[2]],  # 2
        [lower[0], upper[1], lower[2]],  # 3
        [lower[0], lower[1], upper[2]],  # 4
        [upper[0], lower[1], upper[2]],  # 5
        [upper[0], upper[1], upper[2]],  # 6
        [lower[0], upper[1], upper[2]]   # 7
    ])

    faces = [
        [vertices[0], vertices[1], vertices[2], vertices[3]],
        [vertices[4], vertices[5], vertices[6], vertices[7]],
        [vertices[0], vertices[1], vertices[5], vertices[4]],
        [vertices[2], vertices[3], vertices[7], vertices[6]],
        [vertices[1], vertices[2], vertices[6], vertices[5]],
        [vertices[0], vertices[3], vertices[7], vertices[4]]
    ]

    # 添加label到集合
    pc = Poly3DCollection(
        faces, facecolors=color, linewidths=1, edgecolors=color, alpha=alpha
    )
    pc.set_label(label)  # 关键：为集合设置标签
    ax.add_collection3d(pc)
    return pc

# 绘制碰撞区域（红色）和工作空间（蓝色）
collision_cube = plot_cube(ax, cfg.collision_lower_limits, cfg.collision_upper_limits,
                          color='red', alpha=0.2, label='Collision Limits')

# 绘制地下限制平面（灰色）
xx, yy = np.meshgrid(np.linspace(-0.5, 0.5, 2), np.linspace(-0.5, 0.5, 2))
zz = np.full_like(xx, cfg.underground_limit)
underground_plane = ax.plot_surface(xx, yy, zz, color='gray', alpha=0.3, label='Underground Limit')

# 生成球坐标系空间范围的点
r_values = np.linspace(cfg.ranges.init_pos_l[0], cfg.ranges.init_pos_l[1], 20)
theta_values = np.linspace(cfg.ranges.init_pos_p[0], cfg.ranges.init_pos_p[1], 20)
phi_values = np.linspace(cfg.ranges.init_pos_y[0], cfg.ranges.init_pos_y[1], 20)

x_points = []
y_points = []
z_points = []

for r in r_values:
    for theta in theta_values:
        for phi in phi_values:
            x, y, z = sphere2cart(r, theta, phi)
            x_points.append(x)
            y_points.append(y)
            z_points.append(z)

# 绘制球坐标系空间范围的点
ax.scatter(x_points, y_points, z_points, color='blue', alpha=0.1, label='Sphere Range')

# 手动创建图注代理对象（解决Surface图例问题）
from matplotlib.lines import Line2D
legend_proxies = [
    Line2D([0], [0], color='red', lw=4, alpha=0.2, label='Collision Limits'),
    Line2D([0], [0], color='gray', lw=4, alpha=0.3, label='Underground Limit'),
    Line2D([0], [0], color='blue', lw=4, alpha=0.2, label='Sphere Range')
]

# 设置坐标轴和图注
ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('End-Effector Workspace Visualization')
ax.legend(handles=legend_proxies, loc='upper right')  # 使用代理对象

# 调整视角和显示
ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.show()