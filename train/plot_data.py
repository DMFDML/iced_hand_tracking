from matplotlib import pyplot as plt
import numpy as np

m = np.genfromtxt("output_tensors.csv", delimiter=",")

print("[X] min:", m[:, 0].min(), "max:", m[:, 0].max())
print("[Y] min:", m[:, 1].min(), "max:", m[:, 1].max())
print("[Z] min:", m[:, 2].min(), "max:", m[:, 2].max())
print(m.shape)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")
ax.scatter(m[:, 0], m[:, 1], m[:, 2], marker="o")
ax.set_xlabel("X")
ax.set_xlabel("Y")
ax.set_xlabel("Z")
plt.show()
