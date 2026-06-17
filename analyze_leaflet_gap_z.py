from pathlib import Path
import numpy as np

path = Path("/Users/josh/Documents/TEER/mitral_valve.obj")
vertices = []
group_to_faces = {}
current_group = None

with path.open() as f:
    for line in f:
        line = line.strip()
        if line.startswith("v "):
            parts = line.split()
            vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif line.startswith("o ") or line.startswith("g "):
            current_group = line.split()[1]
        elif line.startswith("f "):
            parts = line.split()[1:]
            indices = [int(p.split("/")[0]) - 1 for p in parts]
            if current_group not in group_to_faces:
                group_to_faces[current_group] = []
            group_to_faces[current_group].append(indices)

vertices = np.array(vertices)
g1 = list({idx for face in group_to_faces['ANTERIOR_LEAFLET'] for idx in face})
g2 = list({idx for face in group_to_faces['POSTERIOR_LEAFLET'] for idx in face})

pts1 = vertices[g1]
pts2 = vertices[g2]

# Let's inspect the distance between pts1 and pts2 at different Z levels (e.g. from Z=55.6 to Z=73.9)
# Note that in the original mesh (before we flip it), the perimeter (annulus) is at lower Z or higher Z?
# In _load_clean_obj, we flipped the Z coordinates: v[2] = z_max + z_min - v[2]
# Original Z coordinates are: Z min is ~55.6, Z max is ~73.9.
# Let's check original Z coordinates first.
orig_z1 = pts1[:, 2]
orig_z2 = pts2[:, 2]

# We want to find the distance between pts1 and pts2 at different X and Z coordinates
# Let's write a grid of X and Z, and find the closest point in Y
# Let's print out some stats of Y coordinates for pts1 and pts2
print(f"Anterior leaflet Y range: [{pts1[:, 1].min():.3f}, {pts1[:, 1].max():.3f}]")
print(f"Posterior leaflet Y range: [{pts2[:, 1].min():.3f}, {pts2[:, 1].max():.3f}]")

# Let's plot the average Y coordinate at each X for both leaflets
x_bins = np.linspace(vertices[:, 0].min(), vertices[:, 0].max(), 10)
for i in range(9):
    x_min, x_max = x_bins[i], x_bins[i+1]
    m1 = (pts1[:, 0] >= x_min) & (pts1[:, 0] < x_max)
    m2 = (pts2[:, 0] >= x_min) & (pts2[:, 0] < x_max)
    
    y1_mean = pts1[m1, 1].mean() if np.any(m1) else np.nan
    y2_mean = pts2[m2, 1].mean() if np.any(m2) else np.nan
    print(f"X range [{x_min:.1f}, {x_max:.1f}]: Anterior Y mean = {y1_mean:.3f}, Posterior Y mean = {y2_mean:.3f}, Delta Y mean = {abs(y1_mean - y2_mean):.3f}")
