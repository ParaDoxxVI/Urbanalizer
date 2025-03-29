import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import math


def sequential_touching_circles_with_zero_radius(points):
    """
    Create circles centered at points sequentially, maximizing each radius
    to touch previously created circles. Points that already touch an existing
    circle will have a radius of 0.

    Args:
        points: numpy array of shape (n, 2) containing x,y coordinates

    Returns:
        radii: list of radii for each circle
        order: order in which circles were created
        fig: matplotlib figure with the visualization
    """
    n = len(points)
    radii = np.zeros(n)
    processed = np.zeros(n, dtype=bool)
    order = []

    # Calculate pairwise distances between all points
    dist_matrix = distance_matrix(points, points)
    np.fill_diagonal(dist_matrix, np.inf)

    # Find point with nearest neighbor
    first_point = np.argmin(np.min(dist_matrix, axis=1))
    nearest_to_first = np.argmin(dist_matrix[first_point])

    # Set radius of first circle to half distance to nearest point
    radii[first_point] = dist_matrix[first_point, nearest_to_first] / 2
    processed[first_point] = True
    order.append(first_point)

    # Process remaining points
    while not all(processed):
        # Find nearest unprocessed point to any processed point
        min_dist = np.inf
        next_point = -1
        for i in range(n):
            if not processed[i]:
                # Find minimum distance to any processed point
                for j in order:
                    if dist_matrix[i, j] < min_dist:
                        min_dist = dist_matrix[i, j]
                        next_point = i

        # Check if the point already touches any existing circle
        already_touches = False
        for j in order:
            # Distance between centers
            d = dist_matrix[next_point, j]
            # If distance equals radius of existing circle, it touches exactly
            if abs(d - radii[j]) < 1e-10:  # Using small epsilon for float comparison
                already_touches = True
                break

        if already_touches:
            # Point already touches a circle, give it radius 0
            radii[next_point] = 0
        else:
            # Calculate maximum possible radius for this point
            max_radius = np.inf
            for j in order:
                # Distance between centers
                d = dist_matrix[next_point, j]
                # Max radius so circles touch but don't overlap: r_new = d - r_existing
                possible_radius = d - radii[j]
                max_radius = min(max_radius, possible_radius)

            radii[next_point] = max_radius

        processed[next_point] = True
        order.append(next_point)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot each point and its circle with colors based on order
    cmap = plt.cm.viridis
    colors = [cmap(i / n) for i in range(n)]

    for idx, i in enumerate(order):
        x, y = points[i]

        # Only draw circle if radius > 0
        if radii[i] > 0:
            circle = plt.Circle((x, y), radii[i], fill=False, edgecolor=colors[idx], linewidth=2)
            ax.add_patch(circle)
            ax.plot(x, y, 'o', color=colors[idx], markersize=4)
        else:
            # Mark zero-radius points differently with X
            ax.plot(x, y, 'x', color=colors[idx], markersize=8, mew=2)

        # Add text showing the order
        ax.text(x, y + 1, str(idx + 1), fontsize=8, ha='center', va='center')

    # Set equal aspect ratio
    ax.set_aspect('equal')

    # Set limits with padding
    max_radius = np.max(radii) if np.max(radii) > 0 else 1
    padding = max_radius * 2
    x_min, x_max = np.min(points[:, 0]), np.max(points[:, 0])
    y_min, y_max = np.min(points[:, 1]), np.max(points[:, 1])
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    plt.title('Sequential Circle Packing (X marks points with zero radius)')
    plt.grid(True)

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', markersize=8, label='Point with circle'),
        Line2D([0], [0], marker='x', color='gray', markersize=8, label='Point with zero radius')
    ]
    ax.legend(handles=legend_elements)

    return radii, order, fig


# Example usage
if __name__ == "__main__":
    # Generate some random points
    np.random.seed(42)
    num_points = 20
    points = np.random.rand(num_points, 2) * 100

    radii, order, fig = sequential_touching_circles_with_zero_radius(points)
    plt.show()

    print(f"Circles were created in this order: {[i + 1 for i in order]}")
    print(f"Radii of circles:")
    for i, idx in enumerate(order):
        if radii[idx] > 0:
            print(f"Circle {i + 1} (point {idx}): radius = {radii[idx]:.2f}")
        else:
            print(f"Circle {i + 1} (point {idx}): zero radius (no circle)")