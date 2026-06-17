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

# Let's find for different bins of X coordinate, the minimum distance between pts1 and pts2
x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
print(f"X coordinate range: [{x_min:.3f}, {x_max:.3f}]")

# Let's bin X coordinate into 10 intervals and find the minimum distance in 2D (XY) between anterior and posterior leaflets in each bin
bins = np.linspace(x_min, x_max, 11)
for i in range(10):
    bin_min, bin_max = bins[i], bins[i+1]
    mask1 = (pts1[:, 0] >= bin_min) & (pts1[:, 0] < bin_max)
    mask2 = (pts2[:, 0] >= bin_min) & (pts2[:, 0] < bin_max)
    
    if np.any(mask1) and np.any(mask2):
        p1 = pts1[mask1]
        p2 = pts2[mask2]
        
        # Calculate pairwise distance in XY
        diff = p1[:, np.newaxis, :2] - p2[np.newaxis, :, :2]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        min_dist = np.min(dists)
        print(f"Bin {i} (X in [{bin_min:.1f}, {bin_max:.1f}]): Min XY dist = {min_dist:.3f} mm")
    else:
        print(f"Bin {i} (X in [{bin_min:.1f}, {bin_max:.1f}]): No points")
