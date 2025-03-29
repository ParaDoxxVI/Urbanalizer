import customtkinter as ctk
import shapely
import tkintermapview
import csv
import pathlib
from geopy.geocoders import Nominatim as nmt
from geopy.exc import GeocoderTimedOut
import time
import osmnx as ox
import qwikidata
import qwikidata.sparql
import numpy as np



#activating nominatim's user_agent
geolocator = nmt(user_agent="urbanalizer")

# Calling sensitive information inscribed in a txt file
sens_inf = []
with open("Informations.txt", "r") as fh_txt:
    for line in fh_txt:
        sens_inf.append(str(line).replace("\n", ""))

# Loading geodata from github: https://github.com/bahar/WorldCityLocations/blob/master/World_Cities_Location_table.csv
geodata = []
with open('World_Cities_Location_table.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        geodata.append(line)

root = ctk.CTk()
root.title(f'Urbanalizer by {sens_inf[0]}')
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")  # Fullscreen
root.state("zoomed")  # Maximize on Windows


#---------------- Global Variables -------------------

city_info = []
globalarea = 0
globalpopulation = 0
errors = 0

globalpolygon = []

informationboxcolor = "#212121"
menubarcolor = "#212121"

#---------------- Frontend -------------------

#**Main grids:
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

#**Main frames:
menu = ctk.CTkFrame(master=root, width=root.winfo_screenwidth(), height=round(root.winfo_screenheight()/12), fg_color=menubarcolor, corner_radius=0)
kartenrahmen = ctk.CTkFrame(master=root, width=root.winfo_screenwidth(), height=round((root.winfo_screenheight()*11/12)), fg_color="green", corner_radius=0)
menu.grid(row=0, column=0, sticky="ew")
kartenrahmen.grid(row=1, column=0, sticky="nsew")

#*Menu Bar configuring grid:
menu.grid_columnconfigure(0, weight=1)  # Allow logo to stay on the left
menu.grid_columnconfigure(1, weight=1)  # Push buttons to the right
menu.grid_columnconfigure(2, weight=0)  # Keep world map button flexible
menu.grid_columnconfigure(3, weight=0)  # Keep generator button flexible

#-MENU BAR
logo = ctk.CTkLabel(master=menu, text=" 🏙 URBANALIZER", text_color="white", font=("Bahnschrift", root.winfo_screenheight()/30), justify="left")
logo.grid(row=0, column=0, padx=10, pady=20, sticky="w")

spacer1 = ctk.CTkLabel(master=menu,text="")  # Empty label to create space
spacer1.grid(row=0, column=1, sticky="ew")

#-KARTENRAHMEN
map_widget = tkintermapview.TkinterMapView(master=kartenrahmen, width=kartenrahmen.winfo_screenwidth(), height=kartenrahmen.winfo_screenheight(), padx=0, pady=0)
map_widget.pack(fill="both", expand=True)

watermark = ctk.CTkLabel(master=map_widget, text=f'  ©Open Street Map, Urbanalizer von {sens_inf[0]}, {sens_inf[1]}  ', text_color="white", font=("Bahnschrift", root.winfo_screenheight()/80), justify="left", fg_color="#545454")
watermark.place(relx=1, rely=1, anchor="se")

suchleiste = ctk.CTkFrame(master=map_widget, width=round(kartenrahmen.winfo_screenwidth()*0.3), height=round(kartenrahmen.winfo_screenheight()*0.8), fg_color=informationboxcolor, corner_radius=5)
search_entry = ctk.CTkEntry(master=suchleiste, placeholder_text="Stadt suchen...")
search_button = ctk.CTkButton(master=suchleiste, text="Suchen", command=lambda: search_city())
suchleiste.place(relx=0.01, rely=0.63, anchor="sw")
search_entry.pack(padx=10, pady=10, fill="x")
search_button.pack(padx=10, pady=10)

#-OVERPASS TURBO
big_console_opt=ctk.CTkTextbox(master=kartenrahmen, width=kartenrahmen.winfo_screenwidth(), height=kartenrahmen.winfo_screenheight(), padx=0, pady=0, fg_color="#121212", corner_radius=0,font=("Consolas", root.winfo_screenheight()/60))



def switch_to_overpass_turbo():
    map_widget.pack_forget()
    big_console_opt.pack(fill="both", expand=True)
    big_console_opt.insert("end", str(pathlib.Path(__file__)) + "\n")
    big_console_opt.insert("end", "Overpass Turbo:\n")
    big_console_opt.insert("end", ">>>\n")
    return

def switch_to_weltkarte():
    big_console_opt.delete("0.0", "end")
    big_console_opt.pack_forget()
    map_widget.pack(fill="both", expand=True)




optwaehlen = ctk.CTkButton(master=menu, text="OVERPASS TURBO", text_color="white", font=("Bahnschrift", root.winfo_screenheight()/45), fg_color=menubarcolor, command=lambda:switch_to_overpass_turbo())
optwaehlen.grid(row=0, column=2, padx=10, pady=20, sticky="e")

weltkartewaehlen = ctk.CTkButton(master=menu, text="WELTKARTE", text_color="white", font=("Bahnschrift", root.winfo_screenheight()/45), fg_color=menubarcolor, command=lambda:switch_to_weltkarte())
weltkartewaehlen.grid(row=0, column=3, padx=10, pady=20, sticky="e")





#-INFO_BOX

info_box = ctk.CTkTextbox(
    master=map_widget,
    width=round(map_widget.winfo_screenwidth() * 0.2),
    height=round(map_widget.winfo_screenheight() * 0.72),
    fg_color=informationboxcolor,
    corner_radius=5,
    font=("Bahnschrift", root.winfo_screenheight() / 60)
)
info_box.place(relx=0.99, rely=0.5, anchor="e")
info_box.place_forget()

for i in range(0, 7):
    info_box.insert("end", "\n")

headinginfobox = ctk.CTkTextbox(
        master=info_box,
        width=round(map_widget.winfo_screenwidth() * 0.2),
        height=round(map_widget.winfo_screenheight() * 0.09),
        fg_color=informationboxcolor,
        corner_radius=5,
        font=("Bahnschrift", root.winfo_screenheight() / 45),
        state="disabled"
    )
headinginfobox.place(relx=0.99, rely=0.15, anchor="e")

headinginfobox.configure(state="normal")
headinginfobox.insert("end", " Informationen zu\n", "bold")
headinginfobox.insert("end", " Stadt, Land\n", "bold")
headinginfobox.configure(state="disabled")

close_info_box = ctk.CTkButton(
    master=info_box,
    text="❌",
    font=("Arial", round(info_box.winfo_screenheight() * 0.02)),  # Adjusted size
    text_color="white",
    width=round(info_box.winfo_screenheight() * 0.02),
    command=lambda: thefuncclosingtheinfo_box(),
    fg_color=informationboxcolor
)
close_info_box.place(relx=0.95, rely=0.01, anchor="ne")

visualscaling = ctk.CTkProgressBar(master=info_box, orientation="horizonal")

data_loaded_label = ctk.CTkLabel(master=info_box, text="(Daten ungeladen!)", text_color="Red", font=("Consolas", root.winfo_screenheight()/60))
data_loaded_label.place(relx=0.5, rely=0.9, anchor="s")

kowalski_analysis = ctk.CTkButton(master=info_box, text="Analyse starten", font=("Consolas", root.winfo_screenheight()/60), command=lambda:start_analysis())
kowalski_analysis.place(relx=0.5, rely=0.95, anchor="s")

#-INFOBOX IDLE
blockforreopen = ctk.CTkLabel(master=map_widget, text="", width=round(map_widget.winfo_screenwidth()*0.2), fg_color=informationboxcolor, corner_radius=5,height=round(info_box.winfo_screenheight()*0.04))
reopen_infobox = ctk.CTkButton(master=blockforreopen, text="🟰",font=("Bahnschrift", round(info_box.winfo_screenheight()*0.02)), text_color="white",  fg_color=informationboxcolor, command=lambda:reoppentheinfo_box(), hover_color=informationboxcolor, height=round(info_box.winfo_screenheight()*0.01))


#-CONSOLE BOX

console_box = ctk.CTkTextbox(master=map_widget, width=round(map_widget.winfo_screenwidth()*0.16), height=round(map_widget.winfo_screenheight()*0.3), fg_color="#121212", corner_radius=5, font=("Consolas", root.winfo_screenheight()/60))
console_box.place(relx=0.01, rely=0.99, anchor="sw")
console_box.grid_rowconfigure(0, weight=1)   # Allow row to expand
console_box.grid_columnconfigure(0, weight=1)  # Allow column to expand

console_bar = ctk.CTkLabel(master=console_box,text="Console", fg_color="#333333", corner_radius=5, font=("Consolas", root.winfo_screenheight() / 55), justify="left")
console_bar.grid(row=0, column=0, sticky="new")

console_box.insert("end", "\n")
console_box.insert("end", "\n")

console_box.insert("end", str(pathlib.Path(__file__))+"\n")


# Checkboxes during Analysis

show_city_centers = ctk.CTkCheckBox(
        master=info_box,
        text="Innenstädte und Peripherien anzeigen",
        width=180,
        height=30,
        checkbox_width=20,
        checkbox_height=20,
        corner_radius=5,
        border_width=2,
        fg_color=("#2E3B4E", "#1C252F"),
        border_color=("#607D8B", "#90A4AE"),
        hover_color=("#455A64", "#78909C"),
        text_color=("#E0E0E0", "#FFFFFF"),
        font=("Bahnschrift", 14),
        hover=True
    )

show_public_transport = ctk.CTkCheckBox(
    master=info_box,
    text="ÖPNV-Netz anzeigen",
    width=180,
    height=30,
    checkbox_width=20,
    checkbox_height=20,
    corner_radius=5,
    border_width=2,
    fg_color=("#2E3B4E", "#1C252F"),
    border_color=("#607D8B", "#90A4AE"),
    hover_color=("#455A64", "#78909C"),
    text_color=("#E0E0E0", "#FFFFFF"),
    font=("Bahnschrift", 14),
    hover=True,
)

show_accessibility_zones = ctk.CTkCheckBox(
    master=info_box,
    text="Erreichbarkeitszonen anzeigen",
    width=180,
    height=30,
    checkbox_width=20,
    checkbox_height=20,
    corner_radius=5,
    border_width=2,
    fg_color=("#2E3B4E", "#1C252F"),
    border_color=("#607D8B", "#90A4AE"),
    hover_color=("#455A64", "#78909C"),
    text_color=("#E0E0E0", "#FFFFFF"),
    font=("Bahnschrift", 14),
    hover=True,
)


show_office_buildings = ctk.CTkCheckBox(
    master=info_box,
    text="Bürogebäude anzeigen",
    width=180,
    height=30,
    checkbox_width=20,
    checkbox_height=20,
    corner_radius=5,
    border_width=2,
    fg_color=("#2E3B4E", "#1C252F"),
    border_color=("#607D8B", "#90A4AE"),
    hover_color=("#455A64", "#78909C"),
    text_color=("#E0E0E0", "#FFFFFF"),
    font=("Bahnschrift", 14),
    hover=True,
)


i_want_to_go_back_to_monkey = ctk.CTkButton(master=info_box, text="Zurück", font=("Consolas", root.winfo_screenheight()/60))


#---------------- KEYPRESS ACTIONS -------------------
root.bind("<Return>", lambda event: search_city())

#---------------- FUNCTIONS: -------------------
#**FUNCTIONS FOR PRACTICALITY
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def geodesic_area(vertices):
    from geopy.distance import geodesic
    import numpy as np

    area = 0
    for i in range(len(vertices)):
        j = (i + 1) % len(vertices)
        area += np.radians(vertices[i][1] - vertices[j][1]) * (
                    2 + np.sin(np.radians(vertices[i][0])) + np.sin(np.radians(vertices[j][0])))

    area = round(abs(area * 6371000 * 6371000 / 2)/1000000, 2)
    return area


def throw(text):
    global errors
    console_box.insert("end", "[" + str(errors) +"] " +str(text)+"\n")
    errors +=1
    return

def durchschnittspunkt(coords):
    xs = []
    ys = []
    for coord in coords:
        xs.append(list(coord)[0])
        ys.append(list(coord)[1])
    return (sum(xs)/len(xs), sum(ys)/len(ys))

#**INNER FUNCTIONS:
def callgeocoder(city, country):
    try:
        location = geolocator.geocode(f"{city}, {country}", exactly_one=True)
        if location:
            return location.raw
        else:
            throw(f"Nominatim couldn't find details for {city}, {country}")
            return None
    except GeocoderTimedOut:
        throw("Nominatim request timed out, retrying...")
        time.sleep(1)
        return callgeocoder(city, country)

#**COMMANDS OF WIDGETS
def thefuncclosingtheinfo_box():
    info_box.place_forget()
    close_info_box.place_forget()
    blockforreopen.place(relx=0.99, rely=0.08, anchor="e")
    reopen_infobox.place(relx=0.5, rely=0.5, anchor="center")
    return

def reoppentheinfo_box():
    info_box.place(relx=0.99, rely=0.5, anchor="e")
    close_info_box.place(relx=0.95, rely=0.01, anchor="ne")
    blockforreopen.place_forget()
    reopen_infobox.place_forget()
    return

#---------------------Switch to Overpass Turbo and Map





#----------------------------ANALYSIS FUNCTION COPY PASTE



import geopandas as gpd
import math
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def variance(values, mean):
    return sum([(x-mean)**2 for x in values])

def plotpointsonmap(file, map_widget):
    # Read the GeoJSON file
    gdf = gpd.read_file(file)

    set_of_points = []
    for geom in gdf.geometry:
        if geom.geom_type == 'Point':
            coordinates = (geom.x, geom.y)  # (longitude, latitude)
            set_of_points.append(coordinates)
            marker_2 = map_widget.set_marker(coordinates[1], coordinates[0])

    return set_of_points




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
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

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
    set_of_points = plotpointsonmap(file, map_widget)
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
    maximum_distance = (2*math.pi/360) *6371000*maximum_distance

    radii_mean = np.mean(radii)
    variance_value = variance(radii, radii_mean)
    logarithmic_variance_value = np.log10(variance_value)
    scaled_rating = min(np.floor(-logarithmic_variance_value), 5)

    return scaled_rating, maximum_distance



def evaluate_polycentricity(file, map_widget):
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

    sum = 0
    global globalarea
    area = globalarea
    for radius in radii:
        sum+=(np.pi*((2*math.pi/360) *6371*radius)**2)
    polycentricity = sum/area
    if polycentricity >= 1.4:
        scaled_polyc = 5
    elif polycentricity >= 0.8:
        scaled_polyc = 4
    elif polycentricity >= 0.2:
        scaled_polyc = 3
    elif polycentricity >= 0.08:
        scaled_polyc = 2
    elif polycentricity >= 0.01:
        scaled_polyc = 1
    else:
        scaled_polyc = 0


    return scaled_rating, scaled_polyc




#___________________________________________________


def start_analysis():
    #stadtzentrumermitteln()
    #os.system("analysis_functions.py")


    switchtoanalysis()
    global city_info
    erreichbarkeit = evaluate_transportation(f"Data/{city_info[0]}_{city_info[1]}_TrainStations.geojson", map_widget)
    map_widget.delete_all_marker()
    map_widget.delete_all_polygon()
    # redrawing city boundaries
    drawboundary(str(city_info[0]) + ", " + str(city_info[1]), city_info[0])
    polycentricity_results = evaluate_polycentricity(f"Data/{city_info[0]}_{city_info[1]}_CityElements.geojson", map_widget)


    info_box.configure(state="normal")
    for i in range(0,8):
        info_box.insert("end", "\n")
    info_box.insert("end", f"  Belastungsgleichgewicht der Innenstädte:\n  {int(polycentricity_results[0])}/5 \n")
    info_box.insert("end", f"  Polyzentrität: {polycentricity_results[1]}/5 \n")
    info_box.insert("end", f"  Erreichbarkeit: {int(erreichbarkeit[0])}/5\n")
    info_box.insert("end", f"  Maximale Distanz zu ÖPNV: {round(erreichbarkeit[1])} m \n")

    productivity_rating = int(round(np.mean([polycentricity_results[0], polycentricity_results[1], erreichbarkeit[0]])))

    info_box.insert("end", f"  Produktivität: {productivity_rating} \n")

    info_box.configure(state="disabled")

    return

def command_show_city_centers(value, show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings):
    map_widget.delete_all_marker()
    map_widget.delete_all_polygon()
    # redrawing city boundaries

    uncheck(0, show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings)


    global city_info
    drawboundary(str(city_info[0]) + ", " + str(city_info[1]), city_info[0])
    if value == 1:
        evaluate_polycentricity(f"Data/{city_info[0]}_{city_info[1]}_CityElements.geojson", map_widget)
    else:
        pass
    return

def command_show_transportation(value, show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings):
    map_widget.delete_all_marker()
    map_widget.delete_all_polygon()
    uncheck(1, show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings)
    # redrawing city boundaries
    global city_info
    drawboundary(str(city_info[0]) + ", " + str(city_info[1]), city_info[0])
    if value == 1:
        plotpointsonmap(f"Data/{city_info[0]}_{city_info[1]}_TrainStations.geojson", map_widget)
    else:
        pass
    return

def command_show_accessibility_zones(value, show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings):
    map_widget.delete_all_marker()
    map_widget.delete_all_polygon()
    # redrawing city boundaries
    uncheck(2, show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings)
    global city_info
    drawboundary(str(city_info[0]) + ", " + str(city_info[1]), city_info[0])
    if value == 1:
        evaluate_transportation(f"Data/{city_info[0]}_{city_info[1]}_TrainStations.geojson", map_widget)
    else:
        pass
    return

def command_show_office_buidlings(value, show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings):
    map_widget.delete_all_marker()
    map_widget.delete_all_polygon()
    # redrawing city boundaries
    uncheck(3, show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings)
    global city_info
    drawboundary(str(city_info[0]) + ", " + str(city_info[1]), city_info[0])
    if value == 1:
        plotpointsonmap(f"Data/{city_info[0]}_{city_info[1]}_Offices.geojson", map_widget)
    else:
        pass
    return

def switchtoanalysis():
    info_box.configure(state="normal")
    info_box.delete("0.0", "end")
    info_box.configure(state="disabled")

    headinginfobox.configure(state="normal")
    headinginfobox.delete("0.0", "end")
    headinginfobox.insert("end", "  Analyse:")
    headinginfobox.configure(state="disabled")

    data_loaded_label.place_forget()

    show_city_centers.place(relx=0.05, rely=0.7, anchor="w")
    show_public_transport.place(relx=0.05, rely=0.75, anchor="w")
    show_accessibility_zones.place(relx=0.05, rely=0.80, anchor="w")
    show_office_buildings.place(relx=0.05, rely=0.85, anchor="w")

    i_want_to_go_back_to_monkey.place(relx=0.5, rely=0.95, anchor="s")
    i_want_to_go_back_to_monkey.configure(command=lambda:search_city())


    show_city_centers.configure(command=lambda:command_show_city_centers(show_city_centers.get(), show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings))
    show_public_transport.configure(command=lambda:command_show_transportation(show_public_transport.get(), show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings))
    show_accessibility_zones.configure(command=lambda:command_show_accessibility_zones(show_accessibility_zones.get(), show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings))
    show_office_buildings.configure(command=lambda: command_show_office_buidlings(show_office_buildings.get(), show_city_centers, show_public_transport,show_accessibility_zones, show_office_buildings))

    show_city_centers.select()


    visualscaling.place_forget()
    kowalski_analysis.place_forget()
    return


def uncheck(n, show_city_centers, show_public_transport, show_accessibility_zones, show_office_buildings):


    show_city_centers.select()
    show_public_transport.select()
    show_accessibility_zones.select()
    show_office_buildings.select()

    dict = {0: show_city_centers, 1: show_public_transport, 2: show_accessibility_zones,
            3: show_office_buildings}

    for i in range(0,4):
        if i == n:
            pass
        else:
            dict[i].deselect()
    return

from pathlib import Path

def switchtoinformation(show_city_centers, show_public_transport, show_office_buildings, show_accessibility_zones):




    show_city_centers.place_forget()
    show_public_transport.place_forget()
    show_office_buildings.place_forget()
    show_accessibility_zones.place_forget()

    i_want_to_go_back_to_monkey.place_forget()

    map_widget.delete_all_marker()
    map_widget.delete_all_polygon()

    global city_info
    drawboundary(str(city_info[0]) + ", " + str(city_info[1]), city_info[0])

    data_loaded_label.place(relx=0.5, rely=0.8, anchor="s")
    root.update_idletasks()

    if Path(f'Data/{city_info[0]}_{city_info[1]}_CityElements.txt').is_file() and \
        Path(f'Data/{city_info[0]}_{city_info[1]}_Offices.txt').is_file() and \
        Path(f'Data/{city_info[0]}_{city_info[1]}_TrainStations.txt').is_file():
        data_loaded_label.configure(text="(Daten geladen!)", text_color="Green")
    else:
        data_loaded_label.configure(text="(Daten ungeladen!)", text_color="Red")
    root.update_idletasks()
    kowalski_analysis.place(relx=0.5, rely=0.95, anchor="s")

    return

def inputintokeywords(search):
    temp = str(search)
    temp = temp.split(",")
    counter = 0
    for element in temp:
        temp[counter] = element.strip()
        counter+=1
    return temp

def searchforkeywords(keywords):
    candidates = []
    if len(keywords) !=2:
        throw("Wrong input format (City, Country)")
        return candidates

    for city in geodata:
            if keywords[0] in city[0]:
                if keywords[1] in city[0]:
                    candidates.append(city[0])
    if candidates !=[]:
        return candidates
    else:
        throw("City nonexistent, try writing cities in English or making sure it is recognized as a city. ")

def getcoordinatesofdata(data):
    data[0] = data[0].replace("\"", "")
    data = data[0].split(";")
    coordinates = [0,0]
    coordinates[0] = float(data[-3])
    coordinates[1] = float(data[-2])
    return coordinates

def getnamesofdata(data):
    raw_string = data[0].replace("\"", "")
    info = raw_string.split(";")
    return [info[2], info[1]]

#bundled function for that:
def search_city():


    entry = search_entry.get()
    keywords = inputintokeywords(entry)
    data = searchforkeywords(keywords)

    if not data:
        return  # No city found, exit

    coordinates = getcoordinatesofdata(data)
    names = getnamesofdata(data)

    global city_info
    city_info = names
    map_widget.set_position(coordinates[0], coordinates[1])
    map_widget.set_zoom(13)
    thefuncclosingtheinfo_box()

    switchtoinformation(show_city_centers, show_public_transport, show_office_buildings, show_accessibility_zones)
    print("CLUE")
    headinginfobox.configure(state="normal")
    headinginfobox.delete("0.0", "end")
    headinginfobox.insert("end", " Informationen zu\n", "bold")
    headinginfobox.insert("end", " Stadt, Land\n", "bold")
    headinginfobox.configure(state="disabled")

    headinginfobox.configure(state="normal")
    headinginfobox.delete("2.0", "2.end")
    visualscaling.place_forget()
    headinginfobox.insert("2.0", f" (Eng:) {names[0]}, {names[1]}")
    headinginfobox.configure(state="disabled")

    city_details = callgeocoder(names[0], names[1])
    loadinformationtobox(data, city_details)
    map_widget.delete_all_polygon()


    drawboundary(entry, names[0])
    global globalarea
    info_box.configure(state="normal")
    info_box.insert("end", f' Fläche: {globalarea} km^2\n')
    info_box.configure(state="disabled")
    # Fetch extra city data from Nominatim
    try:
        populationdensity = round(int(globalpopulation)/float(globalarea))
        scaling = populationdensity/50000
        colorscaling = '#%02x%02x%02x' % (255, round(max(255 * (1 - scaling), 0)), round(max(255 * (1 - scaling), 0)))
        info_box.configure(state="normal")
        info_box.insert("end", f" Bev.dichte: {populationdensity} EW/km^2\n")
        info_box.configure(state="disabled")
        visualscaling.configure(progress_color=colorscaling)
        visualscaling.place(relx=0.5, rely=0.5, anchor="center")
        visualscaling.set(min(scaling, 1))

    except:
        throw("Error finding population density.")

    # Update info_box with more details

    global globalpolygon
    print(globalpolygon)

    reoppentheinfo_box()
    return
#city search.


#closing the infobox.

#loading infobox with information of city-----


def loadinformationtobox(data, city_details):
    coordinates = getcoordinatesofdata(data)

    info_box.configure(state="normal")
    info_box.delete("0.0", "end")  # Clear previous content
    for i in range(0,8):
        info_box.insert("end", "\n")
    # Latitude & Longitude
    cardinal1 = "N" if coordinates[0] > 0 else "S"
    cardinal2 = "E" if coordinates[1] > 0 else "W"
    info_box.insert("end", f" ({abs(coordinates[0])}° {cardinal1}, {abs(coordinates[1])}° {cardinal2})\n")
    throw(city_details)
    print(city_details)
    global city_info, cities500
    try:
        wikidataraw = get_city_wikidata(city_info[0], city_info[1])
        population = wikidataraw["population"]["value"]
        global globalpopulation
        globalpopulation = population
    except:
        throw("population not found")
    # If Nominatim found extra details, add them
    if city_details:
        if "display_name" in city_details:
            info_box.insert("end", "(Stadt)\n")
            info_box.insert("end", f" {city_details['display_name']}\n")
        try:
            info_box.insert("end", f" Bevölkerung: {int(population):,d} EW\n".replace(",", "'"))
        except:
            throw("info_box insertion exception at \"Bevoelkerung\"")
        if "extratags" in city_details:
            extratags = city_details["extratags"]
            if "population" in extratags:
                info_box.insert("end", f" Bevölkerung: {population}\n")
            if "area" in extratags:
                info_box.insert("end", f" 📏 Fläche: {extratags['area']} km²\n")

    info_box.configure(state="disabled")



#loading infobox with information of city.

#accessing Nominatim-----
def querygeocoder(query):
    try:
        location = geolocator.geocode(f"{query}", exactly_one=True)
        if location:
            return location.raw  # Returns detailed data
        else:
            throw(f"Nominatim couldn't find details")
            return None
    except GeocoderTimedOut:
        throw("Nominatim request timed out, retrying...")
        time.sleep(1)  # Wait a second before retrying
        return querygeocoder(query)

#accessing Nominatim.

#accessing Osmnx
def getgdffromgeocode(entry):
    boundary=ox.geocode_to_gdf(entry)
    return boundary

def radiantareatokm(radiantarea):
    return ((2*(np.pi)*6350/360)**2)*radiantarea

def drawpolygonfromgdf(gdf, city_name):
    polygons = []
    xs = []
    ys = []
    for multipolygon in gdf["geometry"]:
        if type(multipolygon) == shapely.geometry.polygon.Polygon:
            xs, ys = multipolygon.exterior.xy
            polygon = []
            if len(xs) == len(ys):
                for i in range(0, len(xs)):
                    if i % 10 == 0:
                        polygon.append((ys[i], xs[i]))
                    else:
                        pass
            else:
                throw("Exception: City border polygons invalid.")
            polygons.append(polygon)
        else:

            for geom in multipolygon.geoms:
                xs, ys = geom.exterior.xy
                polygon = []
                if len(xs) == len(ys):
                    for i in range(0, len(xs)):
                        if i % 3 == 0:
                            polygon.append((ys[i], xs[i]))
                        else: pass
                else:
                    throw("Exception: City border polygons invalid.")
                polygons.append(polygon)
    counter = 0
    areas = []
    for polygon in polygons:
        exec(f"polygon{str(counter)} = map_widget.set_polygon(polygon, fill_color=None, outline_color=\"blue\", border_width=5, name=city_name)")
        counter += 1
        matrix = [[],[]]
        for entry in polygon:
            #matrix[0].append(2*np.pi*(entry[0]/360)*6350*np.cos(2*np.pi*(entry[1]/360)))
            matrix[0].append(2 * np.pi * (entry[0] / 360) * 6350)
            matrix[1].append(2*np.pi*(entry[1]/360)*6350)
        areas.append(geodesic_area(polygon))


    global globalarea, globalpolygon
    globalarea = max(areas)
    globalpolygon = polygons[areas.index(globalarea)]




    root.update_idletasks()
    return


def drawboundary(entry, city_name):
    drawpolygonfromgdf(getgdffromgeocode(entry), city_name)
    return


def alias_start_analysis(city_info):
    throw("called for" + str(city_info))
    graph_drive = ox.graph_from_place(f'{city_info[0]}, {city_info[1]}', network_type='drive')
    #graph_walk = ox.graph_from_place(f'{city_info[0]}, {city_info[1]}', network_type='walk')
    #graph_train = ox.graph_from_place('Piedmont, CA, USA', network_type='drive')
    print(type(graph_drive))
    return

def get_city_wikidata(city, country):
    query = """
    SELECT ?city ?cityLabel ?country ?countryLabel ?population
    WHERE
    {
      ?city rdfs:label '%s'@en.
      ?city wdt:P1082 ?population.
      ?city wdt:P17 ?country.
      ?city rdfs:label ?cityLabel.
      ?country rdfs:label ?countryLabel.
      FILTER(LANG(?cityLabel) = "en").
      FILTER(LANG(?countryLabel) = "en").
      FILTER(CONTAINS(?countryLabel, "%s")).
    }
    """ % (city, country)

    res = qwikidata.sparql.return_sparql_query_results(query)
    out = res['results']['bindings'][0]
    return out

def stadtzentrumermitteln():
    global city_info
    query = ["university", "Klinikum", "Museum", "H&M"]
    gebaude = []
    for keyword in query:
        gebaude.append(geolocator.geocode(f"{keyword} {city_info[0]} {city_info[1]}", exactly_one=False))
    counter = 0
    koordinatenaller = []
    for einrichtungen in gebaude:
        for objekt in einrichtungen:
            koordinate = objekt[-1]
            koordinatenaller.append(koordinate)
            exec(f"marker{str(counter)} = map_widget.set_marker(koordinate[0], koordinate[1], text=objekt[0].split(\",\")[0])")
            counter +=1

    map_widget.delete_all_marker()
    point=durchschnittspunkt(koordinatenaller)
    print(point)
    zentrum = map_widget.set_marker(point[0], point[1], text="Zentrum")
    return

import overpy

api = overpy.Overpass()


def overpass_query(occasion, entry):
    city_details = callgeocoder(entry[0], entry[1])
    names = city_details['display_name'].split(", ")
    name = names[0]
    if occasion == "CityElements":
        result = api.query(f"""

        [out:json][timeout:25];
        area["name"~"{name}"](if: t["admin_level"] == "4")->.city;
        (   
            node["shop"="mall"](area.city); 
            way["shop"="mall"](area.city); 
            relation["shop"="mall"](area.city); // Important tourist landmarks

            node["tourism"="attraction"](area.city); 
            way["tourism"="attraction"](area.city); 
            relation["tourism"="attraction"](area.city);

            node["tourism"="hotel"](area.city); 
            way["tourism"="hotel"](area.city); 
            relation["tourism"="hotel"](area.city);
        );
        // Output the results
        out body;
        >;
        out skel qt;


        """)
    elif occasion == "TrainStations":
        result = api.query(f"""

                        [out:json][timeout:25];

                        area["name"~"{name}"](if: t["admin_level"] == "4")->.city;

                        (
                          node["railway"="station"](area.city);
                          way["railway"="station"](area.city);
                          relation["railway"="station"](area.city);

                        );
                        // Output the results
                        out body;
                        >;
                        out skel qt;

                        """)
    elif occasion == "Offices":
        result = api.query(f"""

                [out:json][timeout:25];

                area["name"~"{name}"](if: t["admin_level"] == "4")->.city;
                (
                  node["building"="office"](area.city);
                  way["building"="office"](area.city);
                  relation["building"="office"](area.city);

                );
                // Output the results
                out body;
                >;
                out skel qt;


                """)
    else:
        print("Invalid occasion, please enter one of: [\"city_objects\", \"transportation\", \"offices\"]")
        return
    nodes = result.get_nodes()

    with open(f'Data/{entry[0]}_{entry[1]}_{occasion}.txt', "w") as file:
        file.write(str(nodes))

    nodes = 0

    return


#---------------- DEFAULT SCRIPT: -------------------

map_widget.set_position(52.510885, 13.3989367)
map_widget.set_zoom(12)




#---------------- THE MAGIC WORD: -------------------
root.mainloop()