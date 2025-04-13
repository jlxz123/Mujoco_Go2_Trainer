import mujoco
import mujoco.viewer
import numpy as np
import os

# 设置 MJCF 文件路径
xml_path = "./resources/robots/go2/scene.xml"

# 加载 MJCF 模型
model = mujoco.MjModel.from_xml_path(xml_path)

# 创建仿真数据
data = mujoco.MjData(model)

# 打开可视化窗口
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Press ESC to exit.")
    qpos_pre = np.zeros_like(data.qpos)
    qvel_pre = np.zeros_like(data.qvel)
    # 主循环
    while viewer.is_running():
        action = np.random.rand(12)
        mujoco.mj_step(model, data)
        viewer.sync()
