import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Konstanten
# ------------------------------------------------------------
N = 1200
lmag_voll = 0.333
lmag_gebl = 0.324
invert_B = True

# ------------------------------------------------------------
# Messdaten Vollkern
# ------------------------------------------------------------
I_voll = [
    0.051, 0.248, 0.502, 0.750, 0.999, 1.269, 1.504, 1.770, 2.001,
    2.258, 2.505, 2.749, 3.000, 2.752, 2.499, 2.236, 1.986, 1.745,
    1.499, 1.259, 0.997, 0.747, 0.493, 0.229, 0.049, -0.051, -0.265,
    -0.509, -0.750, -1.006, -1.245, -1.538, -1.754, -2.004, -2.248,
    -2.509, -2.757, -3.003, -2.737, -2.509, -2.227, -1.993, -1.742,
    -1.487, -1.248, -0.989, -0.756, -0.488, -0.238, -0.051, 0.050,
    0.249, 0.547, 0.768, 1.003, 1.246, 1.536, 1.754, 2.010, 2.245,
    2.501, 2.748, 3.000
]

B_voll_raw = [
    -0.325, -0.840, -1.123, -1.347, -1.587, -1.838, -2.060, -2.311,
    -2.520, -2.750, -2.976, -3.198, -3.432, -2.970, -2.581, -2.210,
    -1.890, -1.609, -1.340, -1.094, -0.895, -0.615, -0.377, -0.166,
    -0.029, -0.312, 0.234, 0.423, 0.658, 0.887, 1.077, 1.307, 1.465,
    1.654, 1.826, 2.041, 2.228, 2.432, 1.916, 1.538, 1.117, 0.801,
    0.493, 0.203, -0.052, -0.306, -0.534, -0.787, -1.007, -1.157,
    -0.978, -1.405, -1.621, -1.811, -1.976, -2.129, -2.319, -2.468,
    -2.685, -2.826, -3.016, -3.223, -3.441
]

# ------------------------------------------------------------
# Messdaten geblätterter Kern
# ------------------------------------------------------------
I_gebl = [
    0.050, 0.139, 0.204, 0.292, 0.431, 0.503, 0.611, 0.700, 0.818,
    0.916, 1.000, 0.900, 0.804, 0.696, 0.587, 0.490, 0.403, 0.302,
    0.201, 0.055, 0.050, 0.051, -0.131, -0.205, -0.303, -0.417,
    -0.494, -0.599, -0.697, -0.799, -0.902, -1.002, -0.889, -0.778,
    -0.691, -0.600, -0.494, -0.404, -0.302, -0.205, -0.095, -0.051,
    0.050, 0.115, 0.197, 0.302, 0.401, 0.499, 0.598, 0.699, 0.811,
    0.910, 1.002
]

B_gebl_raw = [
     0.598, -0.761, -1.985, -3.992, -8.186, -10.554, -13.854, -16.205,
    -18.906, -20.585, -21.392, -20.557, -19.178, -17.088, -14.485,
    -11.627, -8.587, -4.875, -1.869, -0.569, 0.628, -0.084, 1.154,
     2.541,  4.789,  8.163, 10.690, 13.956, 16.555, 18.927, 20.791,
    21.781, 20.846, 19.102, 17.372, 15.221, 12.128,  9.025,  5.338,
     2.388,  0.328, -0.167, 0.542, -0.408, -1.867, -4.229, -7.218,
    -10.411, -13.545, -16.247, -18.808, -20.541, -21.438
]

# ------------------------------------------------------------
# Funktionen
# ------------------------------------------------------------
def compute_H(I, N, lmag):
    return np.array(I, dtype=float) * N / lmag

def prepare_B(B_raw, invert=True):
    B = np.array(B_raw, dtype=float)
    return -B if invert else B

def h_zero_crossings(H, B):
    """Interpolierte Schnittpunkte mit H = 0 -> Remanenzpunkte"""
    points = []
    for i in range(len(H) - 1):
        h1, h2 = H[i], H[i + 1]
        b1, b2 = B[i], B[i + 1]

        if h1 == 0:
            points.append((0.0, b1, i))
        elif h1 * h2 < 0:
            b0 = b1 - h1 * (b2 - b1) / (h2 - h1)
            points.append((0.0, b0, i))
    return points

def b_zero_crossings(H, B):
    """Interpolierte Schnittpunkte mit B = 0 -> Koerzitivfeldpunkte"""
    points = []
    for i in range(len(B) - 1):
        b1, b2 = B[i], B[i + 1]
        h1, h2 = H[i], H[i + 1]

        if b1 == 0:
            points.append((h1, 0.0, i))
        elif b1 * b2 < 0:
            h0 = h1 - b1 * (h2 - h1) / (b2 - b1)
            points.append((h0, 0.0, i))
    return points

def select_characteristic_points(plot_name, H, B):
    br_candidates = h_zero_crossings(H, B)
    hc_candidates = b_zero_crossings(H, B)

    # Sättigungspunkt
    idx_sat = int(np.argmax(B))
    Hsat = H[idx_sat]
    Bsat = B[idx_sat]

    if plot_name == "Vollkern":
        # Beide Remanenzpunkte bei H=0
        # -> aus deinen Daten: zwei sinnvolle Schnittpunkte
        br_points = br_candidates[:2]

        # Beide Koerzitivpunkte bei B=0
        # -> aus deinen Daten: genau zwei Schnittpunkte
        hc_points = hc_candidates[:2]

    elif plot_name == "Geblätterter Kern":
        # Beide Remanenzpunkte bei H=0
        br_points = br_candidates[:2]

        # Beim geblätterten Kern entstehen mehrere Nullstellen durch Rauschen.
        # Für Hc verwenden wir:
        # - den größten positiven Schnittpunkt
        # - den betragsgrößten negativen Schnittpunkt
        hc_pos = [p for p in hc_candidates if p[0] > 0]
        hc_neg = [p for p in hc_candidates if p[0] < 0]

        hc_p = max(hc_pos, key=lambda p: p[0]) if hc_pos else None
        hc_n = min(hc_neg, key=lambda p: p[0]) if hc_neg else None

        hc_points = []
        if hc_p is not None:
            hc_points.append(hc_p)
        if hc_n is not None:
            hc_points.append(hc_n)

    else:
        br_points = br_candidates[:2]
        hc_points = hc_candidates[:2]

    # Mittelwerte über die Beträge
    Br_mean = np.mean([abs(p[1]) for p in br_points]) if br_points else None
    Hc_mean = np.mean([abs(p[0]) for p in hc_points]) if hc_points else None

    return br_points, hc_points, Br_mean, Hc_mean, (Hsat, Bsat)

def plot_hysteresis(H, B, title, plot_name):
    br_points, hc_points, Br_mean, Hc_mean, sat_point = select_characteristic_points(plot_name, H, B)
    Hsat, Bsat = sat_point

    plt.figure(figsize=(8, 6))
    plt.plot(
        H, B,
        color='black',
        marker='o',
        markersize=5,
        linewidth=1.8,
        label='Hystereseschleife'
    )

    # Remanenzpunkte markieren
    for i, p in enumerate(br_points):
        plt.scatter(p[0], p[1], color='blue', s=90, zorder=6)

    # Koerzitivpunkte markieren
    for i, p in enumerate(hc_points):
        plt.scatter(p[0], p[1], color='red', s=90, zorder=6)

    # Sättigungspunkt markieren
    plt.scatter(Hsat, Bsat, color='magenta', s=90, zorder=6)

    # Dummy-Einträge für Legende mit gemittelten Werten
    if Br_mean is not None:
        plt.scatter([], [], color='blue',
                    label=fr'$B_r = {Br_mean:.2f}\,\mathrm{{mT}}$')
    if Hc_mean is not None:
        plt.scatter([], [], color='red',
                    label=fr'$H_c = {Hc_mean:.0f}\,\mathrm{{A/m}}$')
    plt.scatter([], [], color='magenta',
                label=fr'$B_{{sat}} = {Bsat:.2f}\,\mathrm{{mT}},\; H_{{sat}} = {Hsat:.0f}\,\mathrm{{A/m}}$')

    plt.axhline(0, color='black', linewidth=1.0)
    plt.axvline(0, color='black', linewidth=1.0)

    plt.xlabel('H in A/m', fontsize=14)
    plt.ylabel('B in mT', fontsize=14)
    plt.title(title, fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='best', framealpha=0.95, fontsize=12)
    plt.tight_layout()
    plt.show()

    print(f"\n--- {title} ---")
    for idx, p in enumerate(br_points, start=1):
        print(f"Br-Punkt {idx}: B = {p[1]:.3f} mT bei H = {p[0]:.1f} A/m")
    if Br_mean is not None:
        print(f"Br (gemittelt) = {Br_mean:.3f} mT")

    for idx, p in enumerate(hc_points, start=1):
        print(f"Hc-Punkt {idx}: H = {p[0]:.3f} A/m bei B = {p[1]:.1f} mT")
    if Hc_mean is not None:
        print(f"Hc (gemittelt) = {Hc_mean:.3f} A/m")

    print(f"Bsat = {Bsat:.3f} mT")
    print(f"Hsat = {Hsat:.3f} A/m")

# ------------------------------------------------------------
# Berechnung
# ------------------------------------------------------------
H_voll = compute_H(I_voll, N, lmag_voll)
B_voll = prepare_B(B_voll_raw, invert=invert_B)

H_gebl = compute_H(I_gebl, N, lmag_gebl)
B_gebl = prepare_B(B_gebl_raw, invert=invert_B)

# ------------------------------------------------------------
# Plots
# ------------------------------------------------------------
plot_hysteresis(H_voll, B_voll, 'Hysteresekurve Vollkern', 'Vollkern')
plot_hysteresis(H_gebl, B_gebl, 'Hysteresekurve geblätterter Kern', 'Geblätterter Kern')