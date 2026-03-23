import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Gitterpunkte (cm)
# -------------------------
x = np.array([-11, -8, -5, -2, 1, 4, 7, 10, 13])
y = np.array([-7, -4, -1, 2, 5, 8])   # aufsteigend sortiert

# -------------------------
# By-Werte (x-Komponente)
# Zeilen entsprechen y = [-7, -4, -1, 2, 5, 8]
# -------------------------
By = np.array([
    [0.09, 0.06, -0.081, -0.442, -0.467, -0.64, 0.152, 0.136, 0.092],      # y = -7
    [0.2, 0.332, 0.464, -1.56, -1.691, 0.582, 0.601, 0.295, 0.156],       # y = -4
    [0.323, 0.841, 3.157, 0.000, 0.000, 5.823, 1.296, 0.451, 0.202],    # y = -1
    [0.302, 0.784, 2.764, 0.000, 0.000, 3.540, 0.987, 0.372, 0.181],    # y = 2
    [0.19, 0.314, 0.295, -0.795, -1.3, -0.125, 0.294, 0.2, 0.122],        # y = 5
    [0.083, 0.078, -0.03, -0.38, -0.365, -0.167, 0.023, 0.061, 0.06]      # y = 8
])

# -------------------------
# Bz-Werte (y-Komponente)
# -------------------------
Bz = np.array([
    [-0.103, -0.274, -0.061, -0.297, 0.371, 0.695, 0.456, 0.265, 0.165],   # y = -7
    [-0.143, -0.478, -1.601, -1.162, 1.267, 2.291, 0.672, 0.257, 0.124],   # y = -4
    [-0.036, -0.283, -1.957, 0.000, 0.000, 0.678, 0.232, 0.042, 0.057],  # y = -1
    [0.187, 0.386, 1.663, 0.000, 0.000, -4.688, -0.747, -0.164, -0.02],  # y = 2
    [0.26, 0.547, 1.281, 1.407, -0.455, -1.471, -0.608, -0.198, -0.052],   # y = 5
    [0.235, 0.357, 0.488, -0.005, -0.094, -0.352, -0.267, -0.128, -0.027]  # y = 8
])

# Betrag
B = np.sqrt(By**2 + Bz**2)

# -------------------------
# Plot
# -------------------------
plt.figure(figsize=(12, 8))

#plt.streamplot(
#    x,
#    y,
#    By,
#    Bz,
 #   color=B,
  #  cmap="viridis",
   # density=1.5,
    #linewidth=1
#)
plt.streamplot(
    x,
    y,
    By,      # x-Komponente (Messwert bleibt unverändert)
    -Bz,     # y-Komponente → Vorzeichenkorrektur
    color=B,
    cmap="viridis_r",
    density=1.5,
    linewidth=1
)
plt.xlim(-11, 13)
plt.ylim(-7, 8)
plt.xticks(np.arange(-11, 14, 2))
plt.yticks(np.arange(-7, 9, 2))
plt.colorbar(label="|B| (mT)")
plt.xlabel("x (cm)")
plt.ylabel("y (cm)")
plt.title("Magnetisches Feld (Streamplot)")
plt.grid(True)

plt.show()