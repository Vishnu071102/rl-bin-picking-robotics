import socket
import time
import numpy as np
from robodk import robolink, robomath
from stable_baselines3 import PPO

# =========================
# LOAD RL MODEL
# =========================
model = PPO.load("rl_bin_picking_model")

# =========================
# ROBO DK SETUP
# =========================
RDK = robolink.Robolink()

robot = RDK.Item('', robolink.ITEM_TYPE_ROBOT)
if not robot.Valid():
    raise Exception("❌ Robot not found")

tool = RDK.Item('', robolink.ITEM_TYPE_TOOL)
if tool.Valid():
    robot.setTool(tool)

# =========================
# SPEED
# =========================
robot.setSpeed(100)

# =========================
# TARGETS
# =========================
home = RDK.Item('Home', robolink.ITEM_TYPE_TARGET)
trigger_pos = RDK.Item('Trigger', robolink.ITEM_TYPE_TARGET)
way_pos = RDK.Item('Waypoint', robolink.ITEM_TYPE_TARGET)

if not home.Valid():
    print("⚠ Home not found → using current pose")
    home = robot.Pose()

if not trigger_pos.Valid():
    raise Exception("❌ Trigger position NOT found")

# =========================
# BIN CONFIG
# =========================
BIN_CENTER = (1042.167, 223.493, -246.000,100,0,0)

BIN_SIZE = {
    "x": 1520,
    "y": 980,
    "z": 690,
    "wall_thickness": 65
}

SAFE_MARGIN = 20

SAFE_BIN = {
    "x_min": BIN_CENTER[0] - BIN_SIZE["x"]/2 + BIN_SIZE["wall_thickness"] + SAFE_MARGIN,
    "x_max": BIN_CENTER[0] + BIN_SIZE["x"]/2 - BIN_SIZE["wall_thickness"] - SAFE_MARGIN,
    "y_min": BIN_CENTER[1] - BIN_SIZE["y"]/2 + BIN_SIZE["wall_thickness"] + SAFE_MARGIN,
    "y_max": BIN_CENTER[1] + BIN_SIZE["y"]/2 - BIN_SIZE["wall_thickness"] - SAFE_MARGIN
}

# =========================
# GRIPPER
# =========================
def gripper_open():
    print("🟢 OPEN")
    time.sleep(1.0)

def gripper_close():
    print("🔴 CLOSE")
    time.sleep(1.0)

# =========================
# TCP (MECH-MIND)
# =========================
def get_mechmind_target():
    HOST = "192.168.56.1"
    PORT = 50000

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(10)
            s.connect((HOST, PORT))

            print("📡 Connected to Mech-Mind")

            # Trigger camera
            s.sendall("101,1,1,0\n".encode())
            s.recv(1024)

            time.sleep(0.5)

            # Get result
            s.sendall("102,1\n".encode())
            response = s.recv(1024).decode().strip()

            print("📥 Raw:", response)

            values = list(map(float, response.split(",")))

            if len(values) < 11:
                return None

            return (
                values[5], values[6], values[7],
                values[8], values[9], values[10]
            )

    except Exception as e:
        print("❌ TCP Error:", e)
        return None

# =========================
# COLLISION
# =========================
def is_collision():
    return RDK.Collisions() > 0

# =========================
# BIN CHECK
# =========================
def is_inside_bin(x, y):
    return (
        SAFE_BIN["x_min"] <= x <= SAFE_BIN["x_max"] and
        SAFE_BIN["y_min"] <= y <= SAFE_BIN["y_max"]
    )

# =========================
# RL PICK
# =========================
def rl_pick(x, y, z, rx, ry, rz):

    state = np.array([x, y, z], dtype=np.float32)
    action, _ = model.predict(state)

    angle, dx, dy = action

    x_new = x + dx
    y_new = y + dy
    rx_new = rx + angle
    approach_z = z + 20

    print(f"🤖 RL → angle={angle:.2f}, dx={dx:.2f}, dy={dy:.2f}")

    if not is_inside_bin(x_new, y_new):
        print("❌ Outside bin → SKIP")
        return False

    approach_pose = robomath.Pose(x_new, y_new, approach_z, rx_new, ry, rz)
    pick_pose = robomath.Pose(x_new, y_new, z, rx_new, ry, rz)

    try:
        robot.MoveJ(approach_pose)

        if is_collision():
            print("❌ Collision → SKIP")
            return False

        robot.MoveL(pick_pose)

        if is_collision():
            print("❌ Collision → SKIP")
            return False

        gripper_close()
        robot.MoveL(approach_pose)
        gripper_open()

        print("✅ Pick success")
        return True

    except Exception as e:
        print("❌ Unreachable → SKIP", e)
        return False

# =========================
# MAIN LOOP
# =========================
if __name__ == "__main__":
    print("🚀 RL BIN PICKING WITH TRIGGER POSITION")

    while True:

        # 1️⃣ HOME
        print("\n🏠 Moving to HOME")
        robot.MoveJ(home)

        # 2️⃣ TRIGGER POSITION
        print("📸 Moving to TRIGGER position")
        robot.MoveJ(trigger_pos)

        # 3️⃣ GET VISION DATA
        target = get_mechmind_target()

        if target:
            print("🎯 Target received")
            rl_pick(*target)
        else:
            print("⚠ No valid data")

        # 4️⃣ BACK TO HOME
        print("🏠 Returning to HOME")
        robot.MoveJ(home)