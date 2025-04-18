import json
import matplotlib.pyplot as plt

# Load the JSON file
with open("results/DQN.json", "r") as f:
    data = json.load(f)

# Split components from each episode
def split_components(episode):
    cart_pos = [step[0] for step in episode]
    pole_pos = [step[1] for step in episode]
    cart_vel = [step[2] for step in episode]
    pole_vel = [step[3] for step in episode]
    return cart_pos, pole_pos, cart_vel, pole_vel

# Extract values
max_cart_pos, max_pole_pos, max_cart_vel, max_pole_vel = split_components(data["max_episode"])
min_cart_pos, min_pole_pos, min_cart_vel, min_pole_vel = split_components(data["min_episode"])

# X-axis steps
max_steps = list(range(len(max_cart_pos)))
min_steps = list(range(len(min_cart_pos)))

# Create combined plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(max_steps, max_cart_pos, label="Max", color='blue')
plt.plot(min_steps, min_cart_pos, label="Min", color='red', linestyle='--')
plt.title("Cart Position")
plt.xlabel("Step")
plt.ylabel("Position")
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(max_steps, max_pole_pos, label="Max", color='blue')
plt.plot(min_steps, min_pole_pos, label="Min", color='red', linestyle='--')
plt.title("Pole Position")
plt.xlabel("Step")
plt.ylabel("Position")
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(max_steps, max_cart_vel, label="Max", color='blue')
plt.plot(min_steps, min_cart_vel, label="Min", color='red', linestyle='--')
plt.title("Cart Velocity")
plt.xlabel("Step")
plt.ylabel("Velocity")
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(max_steps, max_pole_vel, label="Max", color='blue')
plt.plot(min_steps, min_pole_vel, label="Min", color='red', linestyle='--')
plt.title("Pole Velocity")
plt.xlabel("Step")
plt.ylabel("Velocity")
plt.legend()

plt.suptitle("Comparison of Max and Min Episodes", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()
