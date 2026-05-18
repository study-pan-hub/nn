📝 项目概述
本次贡献主要聚焦于两方面：一是为项目中的多个仿真与算法模块（涉及 CARLA、AirSim、MuJoCo、ROS 等平台）补充标准的 requirements.txt 依赖配置文件，大幅降低环境搭建门槛，提升项目可复现性；二是对 Franka Panda 机械臂抓取仿真控制器进行了深度的功能优化与鲁棒性升级，从控制算法、状态机、抓取验证到参数调优，全面增强了系统的抓取成功率与稳定性。

下面重点阐述机械臂抓取项目的技术细节。

🤖 Franka Panda 机械臂抓取仿真：详细技术描述
一、项目背景与目标
该项目基于 MuJoCo 物理引擎，实现了一个 Franka Panda 机械臂的自动抓取与放置仿真。原版控制器（代码二）采用简单的雅可比伪逆位置控制，缺乏姿态控制、抓取验证和重试机制，在立方体初始位置随机偏移时成功率较低。本次优化的目标是：通过引入 6 自由度操作空间控制、抓取接触验证、自动重试以及动态执行器发现，将抓取成功率提升至 95% 以上，并增强系统的鲁棒性。

二、核心算法与技术创新
2.1 6 自由度操作空间控制（位置 + 姿态）
原版控制器仅控制末端执行器的三维位置，导致机械臂在抓取和放置时姿态不固定，影响成功率。新版控制器实现了 6D 雅可比伪逆控制：

同时计算位置雅可比 jp 和姿态雅可比 jr。

使用 mujoco.mju_subQuat 计算目标四元数与当前四元数之间的三维旋转误差。

将位置误差与姿态误差合并为一个 6 维任务误差向量，并构建 6×7 的雅可比矩阵。

通过阻尼最小二乘法（Damped Least Squares）求解关节速度指令，实现平滑且精确的末端位姿跟踪。

python
# 6D 雅可比与控制
jp, jr = self._compute_jacobian()   # 位置和姿态雅可比
if target_quat is not None:
    jacobian = np.vstack([jp, jr])
    task_err = np.concatenate([pos_err, ori_err])
    damping = self.JACOBIAN_DAMPING * np.eye(6)
2.2 抓取接触验证与自动重试机制
原版闭合夹爪后直接抬升，无法判断是否真正抓住物体。新版新增了 PHASE_VERIFY_GRASP 阶段：

在闭合夹爪后，遍历 data.contact 中的所有接触点，检查夹爪几何体与立方体几何体之间是否存在接触。

若检测到接触，则判定抓取成功，进入抬升阶段；否则记录一次失败。

重试次数上限为 3 次，每次失败后重新打开夹爪、调整位置再次尝试。

若所有重试均失败，则放弃任务并返回初始位置，避免无限循环。

python
for i in range(self.data.ncon):
    contact = self.data.contact[i]
    body1 = self.model.body(self.model.geom_bodyid[contact.geom1]).name
    body2 = self.model.body(self.model.geom_bodyid[contact.geom2]).name
    if ("finger" in body1 and "cube" in body2) or ("finger" in body2 and "cube" in body1):
        has_contact = True
2.3 动态夹爪执行器发现
原版硬编码夹爪关节名 finger_joint1、finger_joint2，不兼容不同版本的 Franka 模型。新版通过遍历 model.nu（执行器数量），动态匹配名称中包含 'finger' 或 'gripper' 的执行器 ID，并自动获取其控制范围（ctrlrange），实现通用适配。

python
self.gripper_actuator_ids = []
for i in range(self.model.nu):
    act_name = self.model.actuator(i).name
    if 'finger' in act_name.lower() or 'gripper' in act_name.lower():
        self.gripper_actuator_ids.append(i)
2.4 状态机超时保护与阶段计数
为每个状态阶段增加了独立的步数计数器 _phase_start_step 和超时阈值 max_steps_per_phase。若某一阶段执行时间超过设定步数仍未完成，则强制进入下一阶段，防止因模型卡死或控制参数不当导致的无限循环。

2.5 抓取参数与 PD 增益调优
抓取高度：考虑 Franka 手部中心到指尖的实际距离（约 0.103m），将 grab_height 从 0.05m 调整为 0.11m，使指尖恰好包围立方体中部。

放置位置：从 [0.3, 0.0, 0.1] 改为 [0.4, -0.2, 0.05]，避免与初始位置重叠，同时放置在桌面上方。

PD 增益：提高比例增益 KP 至 300，降低微分增益 KD 至 80，增加力矩限制至 40，增强机械臂在负重抬升时的稳定性。

三、优化效果
指标	原版（代码二）	新版（代码一）	提升
姿态控制	无（固定竖直）	6D 位姿控制	——
抓取接触验证	无	基于碰撞检测的接触验证	——
重试机制	无	最多 3 次自动重试	——
夹爪适配性	硬编码关节名	动态执行器发现	——
防卡死超时保护	无	每阶段超时强制推进	——
四、代码结构变更
新增常量：PHASE_VERIFY_GRASP、max_steps_per_phase、grasp_retries 等。

新增方法：_set_phase_start、_set_phase_start_if_unset、_advance_phase。

重写方法：_move_step 支持可选姿态输入；_grab_phase_machine 增加重试逻辑分支。

删除冗余：移除 LIFT_HEIGHT_INCREMENT，统一使用 safe_lift_height。

📁 其他贡献简要说明
除机械臂项目外，我还为项目中的 10 个功能模块 分别添加了 requirements.txt 依赖配置文件。这些模块涵盖：

CARLA 自动驾驶仿真（静态/动态避障、MPC 控制、数据采集）

AirSim 无人机强化学习（PPO/DQN 训练、迷宫导航、3D 点云扫描）

MuJoCo 人机交互生物力学仿真（肌肉驱动人体模型、ROS 节点）

2D 目标跟踪（CARLA 中的 3D→2D 投影、DeepSORT 特征编码）

TD3 自动驾驶训练（Gymnasium CarRacing-v3 环境）

每个 requirements.txt 均包含核心依赖、可选依赖、安装说明和环境验证命令，显著提升了项目的可复现性和用户上手体验。

本次贡献通过 核心算法优化 与 规范化配置管理，既提升了仿真系统的实用性与稳定性，也为后续开发者提供了清晰的环境指引