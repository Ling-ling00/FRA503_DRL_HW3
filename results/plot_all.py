import json
import matplotlib.pyplot as plt
import os

# === List your JSON files and algorithm names ===
file_info = [
    ("results/A2C.json", "A2C"),
    ("results/DQN.json", "DQN"),
    ("results/Linear_Q.json", "Linear_Q"),
    ("results/MC_REINFORCE.json", "MC_REINFORCE"),
]

# === Store each algorithm's values ===
all_data = {}

def split_components(episode):
    cart_pos = [float(step[0]) for step in episode]
    pole_pos = [float(step[1]) for step in episode]
    cart_vel = [float(step[2]) for step in episode]
    pole_vel = [float(step[3]) for step in episode]
    return cart_pos, pole_pos, cart_vel, pole_vel

# === Load data from each file ===
for file_path, algo_name in file_info:
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            data = json.load(f)
        episode = data["min_episode"]  # or "min_episode"
        all_data[algo_name] = split_components(episode)
    else:
        print(f"[WARN] File not found: {file_path}")

# === Plot ===
plt.figure(figsize=(14, 10))

titles = ["Cart Position", "Pole Position", "Cart Velocity", "Pole Velocity"]

for i, title in enumerate(titles):
    plt.subplot(2, 2, i + 1)
    for algo_name, (cart_pos, pole_pos, cart_vel, pole_vel) in all_data.items():
        y = [cart_pos, pole_pos, cart_vel, pole_vel][i]
        steps = list(range(len(y)))
        plt.plot(steps, y, label=algo_name)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(title)
    plt.legend()

plt.suptitle("Comparison of Algorithms (Min Episodes)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
