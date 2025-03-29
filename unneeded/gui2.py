import tkinter as tk
import customtkinter as ctk
import shapely
import tkintermapview
import csv
import pathlib
from geopy.geocoders import Nominatim as nmt
from geopy.exc import GeocoderTimedOut
import time
import osmnx as ox
import json
import qwikidata
import qwikidata.sparql
import numpy as np
import geopandas as gpd
from shapely.geometry import Polygon
import PIL
from win32con import HELP_SETPOPUP_POS

geolocator = nmt(user_agent="urbanalizer")

with open('cities500.json') as f:
    cities500 = json.load(f)


sens_inf = []
with open('Informations.txt', 'r') as fh_txt:
    for line in fh_txt:
        sens_inf.append(str(line).replace("\n", ""))

root = ctk.CTk()
root.title(f'Urbanalizer by {sens_inf[0]}')
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")  # Fullscreen
root.state("zoomed")  # Maximize on Windows






#-------------------------------------------------------------------------------------------------------
#loading geodata from github:https://github.com/bahar/WorldCityLocations/blob/master/World_Cities_Location_table.csv

geodata = []
with open('World_Cities_Location_table.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        geodata.append(line)

#Global Variables
city_info = [""]
globalarea = 0
globalpopulation = 0
errors=0

informationboxcolor = "#212121"
menubarcolor = "#212121"


#------------------------------------------------------------------------------------------------------
#creating grids:
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)


# Creating frames for organized structure
menu = ctk.CTkFrame(master=root, width=root.winfo_screenwidth(), height=round(root.winfo_screenheight()/12), fg_color=menubarcolor, corner_radius=0)
kartenrahmen = ctk.CTkFrame(master=root, width=root.winfo_screenwidth(), height=round((root.winfo_screenheight()*11/12)), fg_color="green", corner_radius=0)

# (gridding frames)
menu.grid(row=0, column=0, sticky="ew")
kartenrahmen.grid(row=1, column=0, sticky="nsew")

# menu bar --

menu.grid_columnconfigure(0, weight=1)  # Allow logo to stay on the left
menu.grid_columnconfigure(1, weight=1)  # Push buttons to the right
menu.grid_columnconfigure(2, weight=0)  # Keep world map button flexible
menu.grid_columnconfigure(3, weight=0)  # Keep generator button flexible
menu.grid_columnconfigure(4, weight=0)

#logo-
logo = ctk.CTkLabel(master=menu, text=" 🏙 URBANALIZER", text_color="white", font=("Bahnschrift", root.winfo_screenheight()/30), justify="left")
logo.grid(row=0, column=0, padx=10, pady=20, sticky="w")


#Menu spacer-
spacer = ctk.CTkLabel(master=menu,text="")  # Empty label to create space
spacer.grid(row=0, column=1, sticky="ew")
#spacer.grid_forget()

optwaehlen = ctk.CTkButton(master=menu, text="OVERPASS TURBO", text_color="white", font=("Bahnschrift", root.winfo_screenheight()/45), fg_color=menubarcolor)
optwaehlen.grid(row=0, column=2, padx=10, pady=20, sticky="e")


#World Map Button-
weltkartewaehlen = ctk.CTkButton(master=menu, text="WELTKARTE", text_color="white", font=("Bahnschrift", root.winfo_screenheight()/45), fg_color=menubarcolor)
weltkartewaehlen.grid(row=0, column=3, padx=10, pady=20, sticky="e")

#Generator Button-
generator = ctk.CTkButton(master=menu, text="GENERATOR", text_color="White",font=("Bahnschrift", root.winfo_screenheight()/45), fg_color=menubarcolor)
generator.grid(row=0, column=4, padx=10, pady=20, sticky="e")

# kartenrahmen--

# map-
map_widget = tkintermapview.TkinterMapView(master=kartenrahmen, width=kartenrahmen.winfo_screenwidth(), height=kartenrahmen.winfo_screenheight(), padx=0, pady=0)
map_widget.pack(fill="both", expand=True)

# watermark-
watermark = ctk.CTkLabel(master=map_widget, text=f'  ©Open Street Map, Urbanalizer von {sens_inf[0]}, {sens_inf[1]}  ', text_color="white", font=("Bahnschrift", root.winfo_screenheight()/80), justify="left", fg_color="#545454")
watermark.place(relx=1, rely=1, anchor="se")
# Floating search bar-
suchleiste = ctk.CTkFrame(master=map_widget, width=round(kartenrahmen.winfo_screenwidth()*0.3), height=round(kartenrahmen.winfo_screenheight()*0.8), fg_color=informationboxcolor, corner_radius=5)
search_entry = ctk.CTkEntry(master=suchleiste, placeholder_text="Stadt suchen...")
search_button = ctk.CTkButton(master=suchleiste, text="Suchen", command=lambda: search_city())
suchleiste.place(relx=0.01, rely=0.63, anchor="sw")
search_entry.pack(padx=10, pady=10, fill="x")
search_button.pack(padx=10, pady=10)


# Floating information box
info_box = ctk.CTkTextbox(
    master=map_widget,
    width=round(map_widget.winfo_screenwidth() * 0.2),
    height=round(map_widget.winfo_screenheight() * 0.72),
    fg_color=informationboxcolor,
    corner_radius=5,
    font=("Courier New", root.winfo_screenheight() / 60)
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
    font=("Courier New", root.winfo_screenheight() / 45),
    state="disabled"
)
headinginfobox.place(relx=0.99, rely=0.15, anchor="e")

# Insert heading text
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

#laengste_pendelzeit_zeigen =
#opnv_netz_zeigen =
#landwert_zeigen =
#artenvongebäude_zeigen =
#gebiete_zeigen =
#gentrifizierung =
#stadtzentren_zeigen =
#geografische_zeigen =


kowalski_analysis = ctk.CTkButton(master=info_box, text="Analyse starten", font=("Consolas", root.winfo_screenheight()/60), command=lambda:start_analysis())
kowalski_analysis.place(relx=0.5, rely=0.95, anchor="s")

# Add spacing in info_box




blockforreopen = ctk.CTkLabel(master=map_widget, text="", width=round(map_widget.winfo_screenwidth()*0.2), fg_color=informationboxcolor, corner_radius=5,height=round(info_box.winfo_screenheight()*0.04))
reopen_infobox = ctk.CTkButton(master=blockforreopen, text="🟰",font=("Bahnschrift", round(info_box.winfo_screenheight()*0.02)), text_color="white",  fg_color=informationboxcolor, command=lambda:reoppentheinfo_box(), hover_color=informationboxcolor, height=round(info_box.winfo_screenheight()*0.01))

#Floating console box-
console_box = ctk.CTkTextbox(master=map_widget, width=round(map_widget.winfo_screenwidth()*0.16), height=round(map_widget.winfo_screenheight()*0.3), fg_color="#121212", corner_radius=5, font=("Consolas", root.winfo_screenheight()/60))
console_box.place(relx=0.01, rely=0.99, anchor="sw")
console_box.grid_rowconfigure(0, weight=1)   # Allow row to expand
console_box.grid_columnconfigure(0, weight=1)  # Allow column to expand

console_bar = ctk.CTkLabel(master=console_box,text="Console", fg_color="#333333", corner_radius=5, font=("Consolas", root.winfo_screenheight() / 55), justify="left")
console_bar.grid(row=0, column=0, sticky="new")

console_box.insert("end", "\n")
console_box.insert("end", "\n")

console_box.insert("end", str(pathlib.Path(__file__))+"\n")
#console_bar.grid(row=0, fill="x", padx=0, pady=0)


#----------------------------------------------------------------------------------------------------------
#Keypress Actions:
root.bind("<Return>", lambda event: search_city())


#----------------------------------------------------------------------------------------------------------
#FUNCTIONS:
#-----city search:
def PolyArea(x,y):
    return ((891.8)/np.float64(1448.8427619934082))*0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def PolyArea_(x,y):
    a1, a2 = 0, 0
    x.append(x[0])
    y.append(y[0])
    for j in range(len(x) - 1):
        a1 += x[j] * y[j + 1]
        a2 += y[j] * x[j + 1]
    l = abs(a1 - a2) / 2
    return l

def inputintokeywords(search):
    temp = str(search)
    temp = temp.split(",")
    counter = 0
    for element in temp:
        temp[counter] = element.strip()
        counter+=1
    return temp

def throw(text):
    global errors
    console_box.insert("end", "[" + str(errors) +"] " +str(text)+"\n")
    errors +=1
    return

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

    headinginfobox.configure(state="normal")
    headinginfobox.delete("2.0", "2.end")
    visualscaling.place_forget()
    headinginfobox.insert("2.0", f" (Eng:) {names[0]}, {names[1]}")
    headinginfobox.configure(state="disabled")

    city_details = get_city_details(names[0], names[1])
    loadinformationtobox(data, city_details)
    map_widget.delete_all_polygon()


    drawboundary(entry, names[0])
    # Fetch extra city data from Nominatim
    try:
        populationdensity = round(int(globalpopulation)/float(globalarea))
        scaling = populationdensity/50000
        colorscaling = '#%02x%02x%02x' % (255, round(max(255 * (1 - scaling), 0)), round(max(255 * (1 - scaling), 0)))
        visualscaling.configure(progress_color=colorscaling)
        visualscaling.place(relx=0.5, rely=0.5, anchor="center")
        visualscaling.set(min(scaling, 1))
    except:
        throw("Error finding population density.")
    info_box.configure(state="normal")
    info_box.insert("end", f" Bev.dichte: {populationdensity} EW/km^2\n")

    info_box.configure(state="disabled")
    # Update info_box with more details


    reoppentheinfo_box()
    return
#city search.

#closing the infobox-----
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
def get_city_details(city, country):
    try:
        location = geolocator.geocode(f"{city}, {country}", exactly_one=True)
        if location:
            return location.raw  # Returns detailed data
        else:
            throw(f"Nominatim couldn't find details for {city}, {country}")
            return None
    except GeocoderTimedOut:
        throw("Nominatim request timed out, retrying...")
        time.sleep(1)  # Wait a second before retrying
        return get_city_details(city, country)

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
                    if i % 3 == 0:
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
        exec(f"polygon{str(counter)} = map_widget.set_polygon(polygon, fill_color=None, outline_color=\"red\", border_width=5, name=city_name)")
        counter += 1
        matrix = [[],[]]
        for entry in polygon:
            matrix[0].append(2*np.pi*(entry[0]/360)*6350)
            matrix[1].append(2*np.pi*(entry[1]/360)*6350)
        areas.append(PolyArea(matrix[0], matrix[1]))
    global globalarea
    globalarea = areas[0]

    info_box.configure(state="normal")
    info_box.insert("end", f' Fläche: {globalarea} km^2\n')
    info_box.configure(state="disabled")
    root.update_idletasks()
    return


def drawboundary(entry, city_name):
    drawpolygonfromgdf(getgdffromgeocode(entry), city_name)
    return

def start_analysis():
    global city_info
    alias_start_analysis(city_info)
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




#--------------------------------------------------------------------------------------------------------

coordinates=getcoordinatesofdata(searchforkeywords(inputintokeywords("Berlin, Germany")))
map_widget.set_position(coordinates[0], coordinates[1])
map_widget.set_zoom(12)

test = get_city_details("Charite", "Berlin Germany")
print(test)

test2 = get_city_details("Dussmann", "Berlin Mitte Germany")
print(test2)

# Run the Tkinter event loop

root.mainloop()
