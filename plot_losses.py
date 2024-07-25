import re
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import math

# Read the file
run_name = 'mcsm_mnist_central'
path = f'./run/logs/{run_name}/'
filename = path + 'stdout.txt'
with open(filename, 'r') as f:
    data = f.readlines()

timestamps = []
loss_values = []
mcsm_loss_values = []
test_loss_values = []

# Extract relevant data using regular expressions
for line in data:
    timestamp_match = re.search(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
    loss_match = re.search(
        r'loss: ([\d\.]+), mcsm_loss: ([\d\.-]+), test_loss: ([\d\.]+)', line)

    if timestamp_match and loss_match:
        timestamp = datetime.strptime(
            timestamp_match.group(1), '%Y-%m-%d %H:%M:%S,%f')
        timestamps.append(timestamp)

        loss, mcsm_loss, test_loss = loss_match.groups()
        loss_values.append(float(loss))
        mcsm_loss_values.append(float(mcsm_loss))
        test_loss_values.append(float(test_loss))

# Create subplots
fig, axs = plt.subplots(3, sharex=True, figsize=(10, 8))
fig.suptitle('Loss vs Time')

# Plot loss
axs[0].plot(timestamps, loss_values)
axs[0].set_ylabel('Loss')

# Plot mcsm_loss
axs[1].plot(timestamps, mcsm_loss_values)
axs[1].set_ylabel('MCSM Loss')

# Plot test_loss
axs[2].plot(timestamps, test_loss_values)
axs[2].set_ylabel('Test Loss')
axs[2].set_xlabel('Time')

# Format x-axis with date-time
for ax in axs:
    ax.xaxis_date()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.grid(True)

plt.tight_layout()
plt.savefig(f"./assets/{run_name}_loss.png")
plt.close()
