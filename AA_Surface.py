import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D, art3d
from dataclasses import dataclass
from typing import List, Dict, Tuple
from scipy.spatial import KDTree, ConvexHull

@dataclass(frozen=True)
class Atom:
    position: Tuple[float, float, float]
    element: str
    radius: float
    residue: str
    atom_name: str
    color: str
    electronegativity: float

    @property
    def pos_array(self):
        return np.array(self.position)

class AminoAcidAnalyzer:
    SERINE = {
        'CA': {'pos': (0, 0, 0), 'element': 'C'},
        'CB': {'pos': (1.5, 0, 0), 'element': 'C'},
        'OG': {'pos': (1.5, 1.5, 0), 'element': 'O'},
        'N': {'pos': (-1.5, 0, 0), 'element': 'N'},
        'O': {'pos': (0, 1.5, 0), 'element': 'O'}
    }
    
    THREONINE = {
        'CA': {'pos': (4, 0, 0), 'element': 'C'},
        'CB': {'pos': (5.5, 0, 0), 'element': 'C'},
        'OG1': {'pos': (5.5, 1.5, 0), 'element': 'O'},
        'N': {'pos': (2.5, 0, 0), 'element': 'N'},
        'O': {'pos': (4, 1.5, 0), 'element': 'O'},
        'CG2': {'pos': (5.5, -1.5, 0), 'element': 'C'}
    }
    
    ATOM_PROPERTIES = {
        'C': {'radius': 1.70, 'color': '#4682B4', 'electronegativity': 2.55},  # Steel blue
        'N': {'radius': 1.55, 'color': '#FF00FF', 'electronegativity': 3.04},  # Magenta
        'O': {'radius': 1.52, 'color': '#FF4500', 'electronegativity': 3.44},  # Orange-red
        'S': {'radius': 1.80, 'color': '#FFD700', 'electronegativity': 2.58},  # Gold
        'H': {'radius': 1.20, 'color': '#FFFFFF', 'electronegativity': 2.20}   # White
    }

    def __init__(self, probe_radius: float = 1.4, n_points: int = 200):
        self.probe_radius = probe_radius
        self.n_points = n_points

    def generate_sphere_points(self) -> np.ndarray:
        """Generate uniform points on sphere using Golden spiral"""
        points = []
        phi = np.pi * (3.0 - np.sqrt(5.0))
        
        for i in range(self.n_points):
            y = 1 - (i / float(self.n_points - 1)) * 2
            radius = np.sqrt(1 - y * y)
            theta = phi * i
            
            x = np.cos(theta) * radius
            z = np.sin(theta) * radius
            points.append([x, y, z])
            
        return np.array(points)

    def create_amino_acid(self, aa_type: str) -> List[Atom]:
        template = getattr(self, aa_type.upper())
        atoms = []
        
        for atom_name, properties in template.items():
            element = properties['element']
            atom_props = self.ATOM_PROPERTIES[element]
            
            atoms.append(Atom(
                position=properties['pos'],
                element=element,
                radius=atom_props['radius'],
                residue=aa_type,
                atom_name=atom_name,
                color=atom_props['color'],
                electronegativity=atom_props['electronegativity']
            ))
            
        return atoms

    def analyze_surface(self, atoms: List[Atom]) -> Tuple[Dict, np.ndarray]:
        sphere_points = self.generate_sphere_points()
        all_surface_points = []
        exposure_data = {}
        
        positions = np.array([atom.pos_array for atom in atoms])
        kdtree = KDTree(positions)
        
        for i, atom in enumerate(atoms):
            atom_points = sphere_points * (atom.radius + self.probe_radius)
            atom_points = atom_points + atom.pos_array
            
            max_neighbor_dist = atom.radius + 2 * self.probe_radius + max(a.radius for a in atoms)
            neighbors = kdtree.query_ball_point(atom.pos_array, max_neighbor_dist)
            neighbors.remove(i)
            
            exposed_points = []
            for point in atom_points:
                is_exposed = True
                for j in neighbors:
                    other_atom = atoms[j]
                    dist = np.linalg.norm(point - other_atom.pos_array)
                    if dist < (other_atom.radius + self.probe_radius):
                        is_exposed = False
                        break
                
                if is_exposed:
                    exposed_points.append(point)
            
            all_surface_points.extend(exposed_points)
            exposure_data[i] = {
                'atom': atom,
                'total_points': len(atom_points),
                'exposed_points': len(exposed_points),
                'exposure_ratio': len(exposed_points) / len(atom_points)
            }
        
        return exposure_data, np.array(all_surface_points) if all_surface_points else np.zeros((0, 3))

    def visualize_3d_surface(self, atoms: List[Atom], surface_points: np.ndarray, title: str = ""):
        if len(surface_points) < 4:
            print("Not enough surface points for visualization")
            return
        
        # Light background
        fig = plt.figure(figsize=(12, 12), facecolor='#F5F5F5')
        ax = fig.add_subplot(111, projection='3d', facecolor='#F5F5F5')
        ax.set_axis_off()

        # Plot atoms
        for atom in atoms:
            ax.scatter(*atom.position, c=atom.color, s=atom.radius*200, alpha=0.9, 
                      edgecolors='black', linewidth=1)
            ax.text(*atom.position, atom.atom_name, fontsize=10, color='black', 
                    ha='center', va='center', fontweight='bold')

        # 3D surface with smoothed electronegativity gradient
        if len(surface_points) >= 4:
            hull = ConvexHull(surface_points)
            
            # Build KDTree for atoms
            atom_positions = np.array([atom.pos_array for atom in atoms])
            atom_electroneg = np.array([atom.electronegativity for atom in atoms])
            atom_tree = KDTree(atom_positions)
            
            # Query all distances from surface points to atoms (up to a reasonable cutoff)
            cutoff_distance = max(a.radius + self.probe_radius for a in atoms) * 2
            distances, indices = atom_tree.query(surface_points, k=len(atoms), distance_upper_bound=cutoff_distance)
            
            # Smooth electronegativity with inverse distance weighting (IDW)
            surface_electroneg = np.zeros(len(surface_points))
            for i, (dists, idxs) in enumerate(zip(distances, indices)):
                valid_dists = dists[dists < cutoff_distance]  # Only consider points within cutoff
                valid_idxs = idxs[dists < cutoff_distance]
                if len(valid_dists) > 0:
                    weights = 1 / (valid_dists + 1e-6)  # Avoid division by zero
                    surface_electroneg[i] = np.sum(weights * atom_electroneg[valid_idxs]) / np.sum(weights)
                else:
                    # Fallback to nearest atom if no points within cutoff
                    nearest_idx = atom_tree.query(surface_points[i])[1]
                    surface_electroneg[i] = atom_electroneg[nearest_idx]
            
            # Normalize electronegativity for colormap
            vmin, vmax = 2.20, 3.44
            norm_electroneg = (surface_electroneg - vmin) / (vmax - vmin)
            
            # Apply smoothed gradient to hull triangles
            for simplex in hull.simplices:
                vertices = surface_points[simplex]
                vertex_electroneg = norm_electroneg[simplex]
                avg_electroneg = np.mean(vertex_electroneg)
                color = plt.cm.coolwarm(avg_electroneg)
                poly = art3d.Poly3DCollection([vertices])
                poly.set_color(color)
                poly.set_alpha(0.5)
                poly.set_edgecolor('none')
                ax.add_collection3d(poly)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=plt.Normalize(vmin=vmin, vmax=vmax))
            sm.set_array(surface_electroneg)
            cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=5, label='Electronegativity')

        # Adjust view and aspect
        ax.view_init(elev=20, azim=45)
        ax.set_box_aspect([1, 1, 1])
        
        # Set limits with margin
        margin = 2.0
        for i in range(3):
            min_val = min(surface_points[:, i].min(), min(atom.position[i] for atom in atoms))
            max_val = max(surface_points[:, i].max(), max(atom.position[i] for atom in atoms))
            mid_val = (min_val + max_val) / 2
            range_val = max_val - min_val
            ax.set_xlim3d([mid_val - range_val/2 - margin, mid_val + range_val/2 + margin])
            ax.set_ylim3d([mid_val - range_val/2 - margin, mid_val + range_val/2 + margin])
            ax.set_zlim3d([mid_val - range_val/2 - margin, mid_val + range_val/2 + margin])

        plt.title(title, color='black', fontsize=16, pad=20, fontweight='bold')
        plt.tight_layout()
        plt.show()

def analyze_amino_acid_pair():
    analyzer = AminoAcidAnalyzer()
    
    serine = analyzer.create_amino_acid('serine')
    threonine = analyzer.create_amino_acid('threonine')
    
    ser_exposure, ser_surface = analyzer.analyze_surface(serine)
    thr_exposure, thr_surface = analyzer.analyze_surface(threonine)
    combined_exposure, combined_surface = analyzer.analyze_surface(serine + threonine)
    
    print("Visualizing Serine...")
    analyzer.visualize_3d_surface(serine, ser_surface, "Serine Solvent-Accessible Surface")
    
    print("Visualizing Threonine...")
    analyzer.visualize_3d_surface(threonine, thr_surface, "Threonine Solvent-Accessible Surface")
    
    print("Visualizing Combined System...")
    analyzer.visualize_3d_surface(serine + threonine, combined_surface, 
                                 "Combined Amino Acids Solvent-Accessible Surface")
    
    print("\nSurface Analysis Statistics:")
    print("\nSerine Exposure:")
    for idx, data in ser_exposure.items():
        atom = data['atom']
        print(f"{atom.atom_name}: {data['exposure_ratio']:.2%} exposed")
    
    print("\nThreonine Exposure:")
    for idx, data in thr_exposure.items():
        atom = data['atom']
        print(f"{atom.atom_name}: {data['exposure_ratio']:.2%} exposed")
    
    interface_size = (sum(data['exposed_points'] for data in ser_exposure.values()) + 
                      sum(data['exposed_points'] for data in thr_exposure.values()) - 
                      sum(data['exposed_points'] for data in combined_exposure.values()))
    print(f"\nInterface Statistics:")
    print(f"Interface size: {interface_size} points")

if __name__ == "__main__":
    analyze_amino_acid_pair()
