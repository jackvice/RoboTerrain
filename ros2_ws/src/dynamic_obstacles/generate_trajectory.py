import sys
import numpy as np
import collada
from scipy.interpolate import CubicSpline
import xml.etree.ElementTree as ET



def load_terrain_mesh(dae_file):
    """
    Load terrain mesh from a COLLADA (.dae) file using the collada library's
    built-in conveniences for retrieving vertex arrays. Returns x, y, z arrays.
    """
    mesh = collada.Collada(dae_file)
    all_vertices = []

    # Loop over each geometry in the DAE
    for geom in mesh.geometries:
        # Loop over each primitive in the geometry
        for prim in geom.primitives:
            # We're only interested in triangle sets here
            if isinstance(prim, collada.triangleset.TriangleSet):
                # prim.vertex is a NumPy Nx3 array of the actual vertex positions
                # from the triangles in this primitive
                all_vertices.extend(prim.vertex)

    if not all_vertices:
        raise ValueError("No triangle vertices found in the DAE file.")

    # Convert to a NumPy array
    all_vertices = np.array(all_vertices)
    # Remove duplicates if desired
    # The following will ensure each [x,y,z] is unique
    all_vertices = np.unique(all_vertices, axis=0)

    x = all_vertices[:, 0]
    y = all_vertices[:, 1]
    z = all_vertices[:, 2]

    return x, y, z


def create_height_map(x, y, z):
    """Create a 2D height map from terrain vertices."""
    height_map = {}
    for i in range(len(x)):
        height_map[(x[i], y[i])] = z[i]
    return height_map


from scipy.interpolate import griddata

def get_height(height_map, x, y):
    """Get interpolated height for (x, y) using bilinear interpolation."""
    known_points = np.array(list(height_map.keys()))  # (x, y) coordinates
    known_heights = np.array(list(height_map.values()))  # Corresponding z values

    interpolated_height = griddata(known_points, known_heights, (x, y), method='linear')

    if np.isnan(interpolated_height):  # If interpolation fails, use nearest neighbor
        interpolated_height = griddata(known_points, known_heights, (x, y), method='nearest')

    return float(interpolated_height)


def generate_trajectory(waypoints, height_map, velocity=1.0, sample_interval=0.02):
    """Generate a smooth 3D trajectory with constant velocity."""
    waypoints = np.array(waypoints)
    x, y = waypoints[:, 0], waypoints[:, 1]
    z = np.array([get_height(height_map, xi, yi) for xi, yi in zip(x, y)])

    # Compute cumulative distance along the path
    distance = np.cumsum(np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2))
    spline_x, spline_y, spline_z = CubicSpline(distance, x), CubicSpline(distance, y), CubicSpline(distance, z)

    # Resample at regular intervals
    total_distance = distance[-1] + sample_interval
    sampled_distances = np.arange(0, total_distance, sample_interval)
    sampled_x = spline_x(sampled_distances)
    sampled_y = spline_y(sampled_distances)
    sampled_z = spline_z(sampled_distances)

    # Correct yaw computation
    yaw_values = compute_yaw(list(zip(sampled_x, sampled_y, sampled_z)))

    return list(zip(sampled_x, sampled_y, sampled_z, yaw_values))

def compute_yaw(waypoints):
    """Compute yaw (heading) angles for each waypoint."""
    yaw_values = []
    for i in range(len(waypoints) - 1):
        x1, y1, _ = waypoints[i]
        x2, y2, _ = waypoints[i + 1]
        yaw = np.arctan2(y2 - y1, x2 - x1)  # Compute heading direction
        yaw_values.append(yaw)
    
    # Repeat last yaw for the final waypoint to maintain length consistency
    yaw_values.append(yaw_values[-1])
    return yaw_values


def write_sdf(trajectory, output_file):
    """Write trajectory to an SDF file in correct Gazebo format."""
    with open(output_file, "w") as f:
        f.write('<trajectory id="0" type="walk">\n')

        for i, (x, y, z, yaw) in enumerate(trajectory):
            time = f"{i * 0.01:.2f}"  # Ensure small time step (matches working file)
            pose = f"{x} {y} {z} 0 0 {yaw}"
            f.write(f'  <waypoint>\n    <time>{time}</time>\n    <pose>{pose}</pose>\n  </waypoint>\n')

        f.write("</trajectory>\n")
    print(f"Trajectory saved to {output_file}")


def main():
    if len(sys.argv) != 5:
        print("Usage: python script.py <dae_file> velocity x1,y1 x2,y2 x3,y3")
        sys.exit(1)
    
    dae_file = sys.argv[1]
    velocity = sys.argv[2]
    waypoints = [tuple(map(float, arg.split(','))) for arg in sys.argv[2:]]
    
    x, y, z = load_terrain_mesh(dae_file)
    height_map = create_height_map(x, y, z)
    trajectory = generate_trajectory(waypoints, height_map, velocity)
    write_sdf(trajectory, "trajectory_1.sdf")
    
if __name__ == "__main__":
    main()
