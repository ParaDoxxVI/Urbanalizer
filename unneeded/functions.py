
import customtkinter as ctk
import tkintermapview
import geopandas as gpd
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from scipy.spatial import distance_matrix
from math import radians, sin, cos, asin, sqrt

root = ctk.CTk()
root.title(f'Urbanalizer by Test')
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")  # Fullscreen
root.state("zoomed")  # Maximize on Windows

map_widget = tkintermapview.TkinterMapView(master=root, width=root.winfo_screenwidth(), height=root.winfo_screenheight(), padx=0, pady=0)
map_widget.pack(fill="both", expand=True)



def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

def plottransportation(file, map_widget):
    # Read the GeoJSON file
    gdf = gpd.read_file(file)

    set_of_points = []
    for geom in gdf.geometry:
        if geom.geom_type == 'Point':
            coordinates = (geom.x, geom.y)  # (longitude, latitude)
            set_of_points.append(coordinates)
            marker_2 = map_widget.set_marker(coordinates[1], coordinates[0])

    return set_of_points


def scatter_centerbuildings(map_widget):
    gdf = gpd.read_file('../Data/TokioStadtObjekte.geojson')

    set_of_points = []
    # Access coordinates from the geometry column
    for geom in gdf.geometry:
        if geom.geom_type == 'Point':
            coordinates = (geom.x, geom.y)  # (longitude, latitude)
            set_of_points.append(coordinates)
            marker_2 = map_widget.set_marker(coordinates[1], coordinates[0])
        # Handle other geometry types as needed
    print(set_of_points)

    # Find K

    best_k = 10

    # Output the best K value and corresponding Silhouette Score
    print("Best K value:", best_k)

    # K-Means Clustering

    kmeans = KMeans(n_clusters=best_k)
    kmeans.fit(set_of_points)
    x = []
    y = []
    for point in set_of_points:
        x.append(point[0])
        y.append(point[1])
    plt.scatter(x, y, c=kmeans.labels_)
    return

def trimsetofpoints(set_of_points):
    epsilon = 0.01
    iteration = 0
    index = 0
    blacklist = []
    for point in set_of_points:
        index2 = 0
        clustingnumber = 0
        for point2 in set_of_points:
            if index == index2:
                pass
            else:
                if math.dist(point, point2) <= epsilon:
                    clustingnumber += 1
                iteration += 1
            index2 += 1
        if clustingnumber >= 2:
            pass
        else:
            blacklist.append(point)
        index += 1

    newset = list(filter(lambda x: x not in blacklist, set_of_points))
    return newset

def find_clusters_custom(file, map_widget):
    gdf = gpd.read_file(file)

    set_of_points = []
    # Access coordinates from the geometry column
    for geom in gdf.geometry:
        if geom.geom_type == 'Point':
            coordinates = (geom.x, geom.y)  # (longitude, latitude)
            set_of_points.append(coordinates)
        # Handle other geometry types as needed

    set_of_points = trimsetofpoints(set_of_points)
    set_of_points = trimsetofpoints(set_of_points)

    #for point in set_of_points:
        #marker_2 = map_widget.set_marker(point[1], point[0])


    # determine k


    best_k = 20

    # Output the best K value and corresponding Silhouette Score

    # K-Means Clustering

    kmeans = KMeans(n_clusters=best_k)
    kmeans.fit(set_of_points)

    for center in kmeans.cluster_centers_:
        marker_2 = map_widget.set_marker(center[1], center[0])
        drawcircle(center, 0.002, "#d4153e", 2)

    x = []
    y = []
    for point in set_of_points:
        x.append(point[0])
        y.append(point[1])
    plt.scatter(x, y, c=kmeans.labels_)

    cluster_centers = kmeans.cluster_centers_

    return cluster_centers


def drawcircle(pos, r_deg, color, width, name="circle"): #Internetquelle

    center_lon = pos[0]
    center_lat = pos[1]
    vertices = []

    # Convert radius from degrees to radians
    angular_radius = math.radians(r_deg)

    # Number of segments - more for larger circles
    n = max(36, int(2 * math.pi * r_deg * 20))

    for i in range(n):
        bearing = math.radians(i * 360.0 / n)

        # Calculate the latitude of the point
        lat_rad = math.asin(
            math.sin(math.radians(center_lat)) * math.cos(angular_radius) +
            math.cos(math.radians(center_lat)) * math.sin(angular_radius) * math.cos(bearing)
        )

        # Calculate the longitude of the point
        lon_rad = math.radians(center_lon) + math.atan2(
            math.sin(bearing) * math.sin(angular_radius) * math.cos(math.radians(center_lat)),
            math.cos(angular_radius) - math.sin(math.radians(center_lat)) * math.sin(lat_rad)
        )

        # Convert back to degrees
        x = math.degrees(lat_rad)
        y = math.degrees(lon_rad)

        vertices.append((x, y))

    # Close the circle
    vertices.append(vertices[0])

    circle = map_widget.set_polygon(vertices, fill_color=None, outline_color=color, border_width=width, name=name)
    return circle


def haversine_distance(point1, point2):
    """
    Calculate the great-circle distance between two points on a sphere.
    Points are in (longitude, latitude) format in degrees.
    Returns distance in degrees.
    """
    lon1, lat1 = point1
    lon2, lat2 = point2

    # Convert to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))

    # Return angle in degrees
    return c * 180 / np.pi


def geodesic_distance_matrix(points):
    """Create a distance matrix using great-circle distance"""
    n = len(points)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance(points[i], points[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    np.fill_diagonal(dist_matrix, np.inf)
    return dist_matrix


def circlepacking(cluster_centers):
    points = cluster_centers
    n = len(points)
    radii = np.zeros(n)
    processed = np.zeros(n, dtype=bool)
    order = []

    # Calculate pairwise distances between all points
    dist_matrix = geodesic_distance_matrix(points)
    np.fill_diagonal(dist_matrix, np.inf)

    first_point = np.argmin(np.min(dist_matrix, axis=1))
    nearest_to_first = np.argmin(dist_matrix[first_point])

    radii[first_point] = dist_matrix[first_point, nearest_to_first] / 2
    processed[first_point] = True
    order.append(first_point)

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
    dictofpointsandradii = list(zip(points, radii))
    return dictofpointsandradii



def evaluate_transportation(file, map_widget):
    set_of_points = plottransportation(file, map_widget)
    disks = circlepacking(set_of_points)
    radii = []
    for disk in disks:
        radii.append(disk[1])
    # evaluate quantitatively: density, uniformity
    # this might help https://stackoverflow.com/questions/13005294/measure-the-uniformity-of-distribution-of-points-in-a-2d-square
    for disk in disks:
        if disk[1] > 0:
            drawcircle(list(disk[0]), float(disk[1]), "#3589bd", 5)
    else:
        pass
    maximum_distance = max(radii)
    print(maximum_distance)

    return



def evalate_polycentricity(file, map_widget):
    clusters = find_clusters_custom(file, map_widget) #array of points
    #reference_point = (13.3761191, 52.5197306)
    #x = []
    #y = []
    #for i in range(0, 10):
        #dist = 0.004
        #x.append(reference_point[1]+dist*i)
        #y.append(reference_point[0]+dist * i)
    #clusters = list(zip(y, x))


    disks = circlepacking(clusters) # array of (points, radius)
    radii = []
    for disk in disks:
        radii.append(disk[1])


    for disk in disks:
        if disk[1] > 0:
            drawcircle(list(disk[0]), float(disk[1]), "#3589bd", 5)
    else:
        pass


    radii_mean = np.mean(radii)
    variance_value = variance(radii, radii_mean)
    logarithmic_variance_value = np.log10(variance_value)
    scaled_rating = min(np.floor(-logarithmic_variance_value), 5)

    print("Gleichmäßige Belastung der Innenstädte: "+ str(scaled_rating))
    sum = 0
    area = 0
    for radius in radii:
        sum+=(np.pi*radius**2)
    polycentricity = area/sum

    if polycentricity >= 1.2:
        scaled_polyc = 5
    elif polycentricity >= 1:
        scaled_polyc = 4
    elif polycentricity >= 0.8:
        scaled_polyc = 3
    elif polycentricity >= 0.4:
        scaled_polyc = 2
    elif polycentricity >= 0.1:
        scaled_polyc = 1
    else:
        scaled_polyc = 0


    return scaled_rating


def start():
    print(evalate_polycentricity("Data/Stadtobjekte.geojson", map_widget))
    return

start()

root.mainloop()