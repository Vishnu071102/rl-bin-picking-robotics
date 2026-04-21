import gymnasium as gym
from gymnasium import spaces
import numpy as np
from robodk.robolink import *
from robodk.robomath import *
import time  # Added to slow down the view so you can see it


class ABB_BinPickEnv(gym.Env):
    def __init__(self):
        super(ABB_BinPickEnv, self).__init__()
        self.RDK = Robolink()
        self.robot = self.RDK.Item('ABB IRB 4600-60/2.05')

        # Action Space: Small changes to the 6 joint angles [-1.0 to 1.0 degrees]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)

        # Observation Space: 6 Joint Angles + 3 Target Coordinates (X, Y, Z)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        # 1. Move robot to a safe 'Home' position above the bin
        self.robot.MoveJ([0, 0, 0, 0, 90, 0])

        # 2. Randomize a 'Target' inside the bin (Simulating Mech-Mind)
        # Adjust these ranges based on your actual bin location in RoboDK
        self.target_pos = [500 + np.random.uniform(-50, 50),
                           0 + np.random.uniform(-50, 50),
                           200]

        # Return initial observation
        obs = np.concatenate((self.robot.Joints().list(), self.target_pos))
        return obs.astype(np.float32), {}

    def step(self, action):
        # 1. Apply the action (Move joints)
        current_joints = self.robot.Joints().list()
        new_joints = [j + a for j, a in zip(current_joints, action)]
        self.robot.setJoints(new_joints)

        # 2. Calculate Reward
        reward = 0
        terminated = False

        # Check for Collision
        if self.RDK.Collisions() > 0:
            print("BUMP! Collision detected.")
            reward = -100
            terminated = True

        # Check distance to target
        current_pose = self.robot.Pose()
        # Distance calculation
        dist = np.linalg.norm(
            np.array([current_pose.Pos()[0], current_pose.Pos()[1], current_pose.Pos()[2]]) - np.array(self.target_pos))

        if dist < 15:  # If within 15mm of target
            print("SUCCESS! Target reached.")
            reward = 500
            terminated = True
        else:
            reward = -0.1 - (dist / 1000)  # Small penalty for time and distance

        obs = np.concatenate((new_joints, self.target_pos))
        return obs.astype(np.float32), reward, terminated, False, {}


# --- THIS IS WHERE YOU INSERT THE TEST LOOP ---
if __name__ == "__main__":
    print("Starting RoboDK Environment Test...")
    env = ABB_BinPickEnv()
    obs, _ = env.reset()

    for i in range(100):
        random_action = env.action_space.sample()  # The AI "guessing"
        obs, reward, done, _, _ = env.step(random_action)

        # Optional: Add a tiny sleep so you can actually watch the robot move
        # time.sleep(0.01)

        if done:
            print(f"Episode finished. Reward: {reward}. Resetting...")
            obs, _ = env.reset()

    print("Test complete. The bridge is working!")