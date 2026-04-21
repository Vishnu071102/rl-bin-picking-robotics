import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO

# =========================
# CUSTOM ENVIRONMENT
# =========================
class BinPickingEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # State: target x,y,z
        self.observation_space = spaces.Box(low=-1000, high=1000, shape=(3,), dtype=np.float32)

        # Action: angle, dx, dy
        self.action_space = spaces.Box(
            low=np.array([-30, -50, -50]),
            high=np.array([30, 50, 50]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.target = np.random.uniform([-500, -300, 0], [500, 300, 500])
        return self.target.astype(np.float32), {}

    def step(self, action):
        angle, dx, dy = action

        # Simulated collision logic (simple model)
        collision = False

        if abs(self.target[0] + dx) > 600:
            collision = True
        if abs(self.target[1] + dy) > 400:
            collision = True
        if abs(angle) > 25:
            collision = True

        # Reward
        if collision:
            reward = -10
            done = True
        else:
            reward = 10
            done = True

        return self.target.astype(np.float32), reward, done, False, {}

# =========================
# TRAIN MODEL
# =========================
env = BinPickingEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64
)

model.learn(total_timesteps=50000)

model.save("rl_bin_picking_model")