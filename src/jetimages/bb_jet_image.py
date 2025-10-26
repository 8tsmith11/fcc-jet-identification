import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import vector
import os
from tqdm import tqdm

# === SETTINGS ===
filename = "/data/users/vcalvo/Pythia_root_files/p8_ZH_ecm240/p8_ee_ZH_ecm240_edm4hep.root"
output_dir_b = "/data/users/vcalvo/jet_image_test/bb_jet_images"
output_dir_nonb = "/data/users/vcalvo/jet_image_test/nonb_jet_images"
n_eta_bins = 4
n_phi_bins = 4
max_b_jets = 100
max_nonb_jets = 100
eta_range = (-3, 3)
phi_range = (-np.pi, np.pi)

os.makedirs(output_dir_b, exist_ok=True)
os.makedirs(output_dir_nonb, exist_ok=True)

# === REGISTER VECTOR BEHAVIORS WITH AWKWARD ===
vector.register_awkward()

# === LOAD ROOT FILE ===
f = uproot.open(filename)
t = f["events"]

# --- Particle data ---
px = t["Particle.momentum.x"].array()
py = t["Particle.momentum.y"].array()
pz = t["Particle.momentum.z"].array()
mass = t["Particle.mass"].array()
pdg = t["Particle/Particle.PDG"].array()  # For identifying b-quarks
part_E = np.sqrt(px**2 + py**2 + pz**2 + mass**2)

particles = vector.awk(
    ak.zip({
        "px": px,
        "py": py,
        "pz": pz,
        "E": part_E
    }, with_name="Momentum4D")
)

# --- Jet data ---
jet_px = t["Jet.momentum.x"].array()
jet_py = t["Jet.momentum.y"].array()
jet_pz = t["Jet.momentum.z"].array()
jet_E = t["Jet.energy"].array()

jets = vector.awk(
    ak.zip({
        "px": jet_px,
        "py": jet_py,
        "pz": jet_pz,
        "E": jet_E
    }, with_name="Momentum4D")
)

# --- Jet–particle mapping ---
jet_constituents = t["_Jet_particles"].array()  # Records with '_Jet_particles.index'

# === HISTOGRAM BINS ===
eta_bins = np.linspace(*eta_range, n_eta_bins + 1)
phi_bins = np.linspace(*phi_range, n_phi_bins + 1)

n_saved_b = 0
n_saved_nonb = 0

# === LOOP OVER EVENTS AND JETS ===
for event_idx in tqdm(range(len(jets)), desc="Processing jets"):
    jets_in_event = jets[event_idx]
    all_const_idx = jet_constituents[event_idx]["_Jet_particles.index"]

    for j, jet in enumerate(jets_in_event):
        # Stop if both limits reached
        if n_saved_b >= max_b_jets and n_saved_nonb >= max_nonb_jets:
            break

        if j >= len(all_const_idx):
            continue
        jet_const_idx = all_const_idx[j]

        if isinstance(jet_const_idx, (int, np.integer)):
            jet_const_idx = [jet_const_idx]
        if not jet_const_idx:
            continue

        # Identify b-jet
        is_b_jet = any(abs(pdg[event_idx][idx]) == 5 for idx in jet_const_idx)

        # Decide output folder and counter
        if is_b_jet and n_saved_b < max_b_jets:
            out_dir = output_dir_b
            n_saved = n_saved_b
            n_saved_b += 1
        elif not is_b_jet and n_saved_nonb < max_nonb_jets:
            out_dir = output_dir_nonb
            n_saved = n_saved_nonb
            n_saved_nonb += 1
        else:
            continue  # Skip if we reached max for this class

        # Particle vectors
        parts = particles[event_idx][jet_const_idx]
        eta_rel = parts.eta - jet.eta
        phi_rel = parts.phi - jet.phi
        pt = parts.pt
        phi_rel = (phi_rel + np.pi) % (2 * np.pi) - np.pi

        # 2D histogram
        H, _, _ = np.histogram2d(
            ak.to_numpy(eta_rel),
            ak.to_numpy(phi_rel),
            bins=(eta_bins, phi_bins),
            weights=ak.to_numpy(pt)
        )

        if H.max() > 0:
            H /= H.max()

        # Save figure
        plt.figure(figsize=(3, 3))
        plt.imshow(H.T, origin="lower", cmap="inferno",
                   extent=[eta_range[0], eta_range[1], phi_range[0], phi_range[1]])
        plt.colorbar(label="pT fraction")
        plt.xlabel(r"$\Delta\eta$")
        plt.ylabel(r"$\Delta\phi$")
        plt.title(f"Event {event_idx}, Jet {j} ({n_eta_bins}×{n_phi_bins})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"jet_{n_saved:04d}.png"), dpi=100)
        plt.close()

    if n_saved_b >= max_b_jets and n_saved_nonb >= max_nonb_jets:
        break

print(f"\n✅ Saved {n_saved_b} b-jet images in '{output_dir_b}'")
print(f"✅ Saved {n_saved_nonb} non-b-jet images in '{output_dir_nonb}'")
