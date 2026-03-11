import numpy as np
import matplotlib.pyplot as plt

# Create figure and axes
fig, ax = plt.subplots(figsize=(10, 6))

# Generate x values (positive values only, as log is defined for x > 0)
x = np.linspace(0, 3, 400)

# Plot different logarithmic functions
ax.plot(x, np.log(x), "r-", linewidth=2, label="ln(x)")  # Natural logarithm

# Add labels and title
ax.set_xlabel("x", fontsize=12)
ax.set_ylabel("log(x)", fontsize=12)
ax.set_title(
    "Logarithmic Function", fontsize=14, fontweight="bold"
)

# Add grid
ax.grid(True, alpha=0.3, linestyle="--")

# Add legend
ax.legend(fontsize=11)

# Set axis limits
ax.set_xlim(0, 3)
ax.set_ylim(-5, 3)

# Add reference lines
ax.axhline(y=0, color="k", linestyle="-", alpha=0.2, linewidth=0.8)
ax.axvline(x=1, color="k", linestyle="-", alpha=0.2, linewidth=0.8)

# Display the plot
plt.tight_layout()
plt.show()

