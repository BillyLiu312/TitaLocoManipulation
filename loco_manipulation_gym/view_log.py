import numpy as np
import matplotlib.pyplot as plt

# 加载单个episode的数据
data = np.load("logs/episode_0001.npz")

# 查看文件中包含的数组
print("文件中包含的数据键:", list(data.keys()))

# 访问观测数据
obs = data['obs']
print(f"原始观测数据形状: {obs.shape}")  # 通常是 (timesteps, num_envs, obs_dim) 或 (num_envs, timesteps, obs_dim)

# 提取第一个环境的所有观测数据
env_idx = 0  # 第一个环境的索引
if len(obs.shape) == 3:
    # 假设形状为 (timesteps, num_envs, obs_dim)
    obs_env0 = obs[:, env_idx, :]
elif len(obs.shape) == 2:
    # 可能是 (timesteps, obs_dim) 单环境情况
    obs_env0 = obs
else:
    raise ValueError(f"意外的观测数据形状: {obs.shape}")

print(f"\n第一个环境的观测数据形状: {obs_env0.shape}")  # 应该是 (timesteps, obs_dim)

# 查看前几条观测数据
print("\n前5个时间步的观测数据:")
for i in range(min(5, obs_env0.shape[0])):
    print(f"时间步 {i}: {obs_env0[i]}")

# 观测数据统计信息
print("\n观测数据统计信息:")
print(f"各维度平均值: {np.mean(obs_env0, axis=0)}")
print(f"各维度标准差: {np.std(obs_env0, axis=0)}")
print(f"各维度最小值: {np.min(obs_env0, axis=0)}")
print(f"各维度最大值: {np.max(obs_env0, axis=0)}")

# 创建时间步数组
timesteps = np.arange(obs_env0.shape[0])

# 绘制各观测维度随时间变化
plt.figure(figsize=(12, 8))
for dim in range(obs_env0.shape[1]):
    plt.plot(timesteps, obs_env0[:, dim], label=f'Obs dim {dim}')
plt.title(f"Observation Dimensions Over Time (Env {env_idx})")
plt.xlabel("Time step")
plt.ylabel("Observation value")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()

# 也可以选择只绘制感兴趣的特定观测维度
interesting_dims = [0, 1, 2, 3]  # 示例：选择前4个维度
plt.figure(figsize=(12, 6))
for dim in interesting_dims:
    plt.plot(timesteps, obs_env0[:, dim], label=f'Obs dim {dim}')
plt.title(f"Selected Observation Dimensions Over Time (Env {env_idx})")
plt.xlabel("Time step")
plt.ylabel("Observation value")
plt.legend()
plt.grid(True)
plt.show()

# 观测维度相关性热力图
if obs_env0.shape[1] > 1:  # 只有多个观测维度时才绘制
    corr_matrix = np.corrcoef(obs_env0, rowvar=False)
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Observation Dimensions Correlation Matrix")
    plt.xticks(np.arange(obs_env0.shape[1]), labels=[f'dim{i}' for i in range(obs_env0.shape[1])])
    plt.yticks(np.arange(obs_env0.shape[1]), labels=[f'dim{i}' for i in range(obs_env0.shape[1])])
    plt.show()