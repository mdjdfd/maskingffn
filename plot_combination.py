import matplotlib.pyplot as plt
import numpy as np
import os

DPI = 1200

prune_iterations = 10

d = np.load(f"{os.getcwd()}/dumps/lt_compression.dat", allow_pickle=True)
b = np.load(f"{os.getcwd()}/dumps/lt_bestaccuracy.dat", allow_pickle=True)
c = np.load(f"{os.getcwd()}/dumps/reinit_bestaccuracy.dat", allow_pickle=True)

a = np.arange(prune_iterations)
plt.plot(a, b, c="blue", label="Winning tickets")
plt.plot(a, c, c="red", label="Random reinit")
plt.xlabel("Weights %")
plt.ylabel("Test accuracy")
plt.xticks(a, d, rotation="vertical")
plt.ylim(0, 100)
plt.legend()
plt.grid(color="gray")

plt.savefig(f"{os.getcwd()}/plots/combined_fc1_mnist.png", dpi=DPI, bbox_inches='tight')
plt.close()
