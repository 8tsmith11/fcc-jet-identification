import uproot
import numpy as np
import matplotlib.pyplot as plt
import fastjet as fj
import os
from math import pi


INPUT_FILE = "/data/users/vcalvo/Pythia_root_files/p8_ZH_mumu_bb_ecm240/p8_ee_ZH_ecm240_edm4hep.root"
OUTPUT_DIR = "/data/users/vcalvo/jet_image_test/jet_test_bb"
R_PARAM = 0.6                      # anti-kt jet radius
PT_MIN = 10.0                      # minimum jet pT [GeV]
IMG_SIZE = 4                       # 4×4 image
IMG_EXTENT = 0.8                   # Δη × Δφ window
MATCH_DR = 0.3                     # ΔR match threshold for b-quarks

os.makedirs(OUTPUT_DIR, exist_ok=True)


# Helper functions
# -------------------------
def make_jet_image(particles, img_size=4, img_extent=0.8):
    etas = np.array([p.eta() for p in particles])
    phis = np.array([p.phi() for p in particles])
    pts = np.array([p.pt() for p in particles])
    eta_cent = np.average(etas, weights=pts)
    phi_cent = np.angle(np.average(np.exp(1j * phis), weights=pts))
    deta = etas - eta_cent
    dphi = (phis - phi_cent + pi) % (2 * pi) - pi
    half = img_extent / 2
    xedges = np.linspace(-half, half, img_size + 1)
    yedges = np.linspace(-half, half, img_size + 1)
    H, _, _ = np.histogram2d(deta, dphi, bins=[xedges, yedges], weights=pts)
    H /= (pts.sum() + 1e-12)
    return H

def make_pseudojets(px, py, pz, E):
    return [fj.PseudoJet(float(px[i]), float(py[i]), float(pz[i]), float(E[i]))
            for i in range(len(px))]


file = uproot.open(INPUT_FILE)
tree = file["events"]

# Reconstructed particles (these have energy)
px_reco = tree["ReconstructedParticles.momentum.x"].array()
py_reco = tree["ReconstructedParticles.momentum.y"].array()
pz_reco = tree["ReconstructedParticles.momentum.z"].array()
E_reco  = tree["ReconstructedParticles.energy"].array()

# Generator-level particles (no energy branch → we’ll compute it)
pdg = tree["Particle.PDG"].array()
px_gen = tree["Particle.momentum.x"].array()
py_gen = tree["Particle.momentum.y"].array()
pz_gen = tree["Particle.momentum.z"].array()

# PDG mass table (in GeV)
PDG_MASS = {
    5: 4.18,    -5: 4.18,
    11: 0.000511,  -11: 0.000511,
    13: 0.10566,   -13: 0.10566,
    23: 91.1876,   # Z
    25: 125.0      # Higgs
}

def mass_from_pdg(pid):
    return PDG_MASS.get(int(pid), 0.0)

# Compute generator-level energy
E_gen = []
for iev in range(len(px_gen)):
    m_arr = np.array([mass_from_pdg(pid) for pid in pdg[iev]])
    E_evt = np.sqrt(px_gen[iev]**2 + py_gen[iev]**2 + pz_gen[iev]**2 + m_arr**2)
    E_gen.append(E_evt)


# FastJet definition
# -------------------------
jetdef = fj.JetDefinition(fj.antikt_algorithm, R_PARAM)


# Loop over events
# -------------------------
for iev in range(len(px_reco)):
    # Reconstructed particles → pseudojets
    particles = make_pseudojets(px_reco[iev], py_reco[iev], pz_reco[iev], E_reco[iev])
    if len(particles) == 0:
        continue

    # Cluster jets
    cs = fj.ClusterSequence(particles, jetdef)
    jets = [j for j in fj.sorted_by_pt(cs.inclusive_jets(PT_MIN)) if j.pt() > PT_MIN]

    # Find generator-level b quarks
    b_idx = np.where(np.abs(pdg[iev]) == 5)[0]
    if len(b_idx) == 0:
        continue
    b_quarks = make_pseudojets(px_gen[iev][b_idx], py_gen[iev][b_idx],
                               pz_gen[iev][b_idx], E_gen[iev][b_idx])

    # Match jets to b-quarks (ΔR < 0.3)
    for jidx, jet in enumerate(jets):
        if any(jet.delta_R(bq) < MATCH_DR for bq in b_quarks):
            # Build and save 4x4 image
            H = make_jet_image(jet.constituents(), img_size=IMG_SIZE, img_extent=IMG_EXTENT)
            outpath = os.path.join(OUTPUT_DIR, f"event{iev}_jet{jidx}_bjet_4x4.png")
            plt.imshow(H.T, origin="lower",
                       extent=[-IMG_EXTENT/2, IMG_EXTENT/2, -IMG_EXTENT/2, IMG_EXTENT/2],
                       aspect="equal")
            plt.xlabel("Δη")
            plt.ylabel("Δφ")
            plt.title(f"b-jet image (event {iev}, jet {jidx})")
            plt.colorbar(label="pT fraction")
            plt.tight_layout()
            plt.savefig(outpath, dpi=150)
            plt.close()

print("✅ Done! b-jet 4×4 images saved in:", OUTPUT_DIR)
