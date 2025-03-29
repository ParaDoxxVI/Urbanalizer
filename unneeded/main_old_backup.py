import tkinter as tk
import customtkinter as ctk
import tkintermapview
import csv

root = ctk.CTk()
root.title("Urban Planning Map")
root.geometry(f"{root.winfo_screenwidth()}x{root.winfo_screenheight()}+0+0")  # Fullscreen
root.state("zoomed")  # Maximize on Windows

#-------------------------------------------------------------------------------------------------------
#loading geodata from github:https://github.com/bahar/WorldCityLocations/blob/master/World_Cities_Location_table.csv

geodata = []
with open('World_Cities_Location_table.csv', mode='r') as file:
    csvFile = csv.reader(file)
    for line in csvFile:
        geodata.append(line)

#------------------------------------------------------------------------------------------------------
#creating grids:
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)


# Creating frames for organized structure
menu = ctk.CTkFrame(master=root, width=root.winfo_screenwidth(), height=round(root.winfo_screenheight()/12), fg_color="#212121", corner_radius=0)
kartenrahmen = ctk.CTkFrame(master=root, width=root.winfo_screenwidth(), height=round((root.winfo_screenheight()*11/12)), fg_color="green", corner_radius=0)

# (gridding frames)
menu.grid(row=0, column=0, sticky="ew")
kartenrahmen.grid(row=1, column=0, sticky="nsew")

# menu bar --
#logo-
logo = ctk.CTkLabel(master=menu, text=" 🏙 URBANALIZER", text_color="white", font=("Bahnschrift", root.winfo_screenheight()/30), justify="left")
logo.grid(row=0, column=0, padx=10, pady=20, sticky="w")

#World Map Button-
weltkartewaehlen = ctk.CTkButton(master=menu, text="WELTKARTE")
weltkartewaehlen.grid(row=0, column=1, padx=10, pady=20, sticky="w")

#Generator Button-
generator = ctk.CTkButton(master=menu, text="GENERATOR")
generator.grid(row=0, column=2, padx=10, pady=20, sticky="w")

# kartenrahmen--

# map-
map_widget = tkintermapview.TkinterMapView(master=kartenrahmen, width=kartenrahmen.winfo_screenwidth(), height=kartenrahmen.winfo_screenheight(), padx=0, pady=0)
map_widget.pack(fill="both", expand=True)

# watermark-
watermark = ctk.CTkLabel(master=map_widget, text="  ©Open Street Map, Urbanalizer von Hikari Nishimoto, Heinrich-Hertz-Gymnasium, 5. PK Geografie - 2025  ", text_color="white", font=("Bahnschrift", root.winfo_screenheight()/80), justify="left", fg_color="#545454")
watermark.place(relx=1, rely=1, anchor="se")

# Floating search bar-
suchleiste = ctk.CTkFrame(master=map_widget, width=round(kartenrahmen.winfo_screenwidth()*0.3), height=round(kartenrahmen.winfo_screenheight()*0.8), fg_color="#333333", corner_radius=5)
search_entry = ctk.CTkEntry(master=suchleiste, placeholder_text="Stadt suchen...")
search_button = ctk.CTkButton(master=suchleiste, text="Suchen", command=lambda: search_city())
suchleiste.place(relx=0.01, rely=0.99, anchor="sw")  # Move closer to bottom left
search_entry.pack(padx=10, pady=10, fill="x")
search_button.pack(padx=10, pady=5)

# Floating information box-
info_box = ctk.CTkTextbox(master=map_widget, width=round(map_widget.winfo_screenwidth()*0.2), height=round(map_widget.winfo_screenheight()*0.8), fg_color="#444444", corner_radius=5, font=("Bahnschrift", root.winfo_screenheight()/45))
info_box.place(relx=0.99, rely=0.5, anchor="e")  # Positioned on the right side
info_box.insert("end", "Informationen zu\n", "bold")
info_box.insert("end", "Stadt, Land\n", "bold")


#----------------------------------------------------------------------------------------------------------
#FUNCTIONS:
#-----city search:
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
    for city in geodata:
        if keywords[0] in city[0]:
            if keywords[1] in city[0]:
                candidates.append(city[0])
    return candidates

def getcoordinatesofdata(data):
    data[0] = data[0].replace("\"", "")
    data = data[0].split(";")
    coordinates = [0,0]
    coordinates[0] = float(data[-3])
    coordinates[1] = float(data[-2])
    return coordinates

#bundled function for that:
def search_city():
    entry=search_entry.get()
    coordinates=getcoordinatesofdata(searchforkeywords(inputintokeywords(entry)))
    map_widget.set_position(coordinates[0], coordinates[1])
    map_widget.set_zoom(13)
    return

#city search.
coordinates=getcoordinatesofdata(searchforkeywords(inputintokeywords("Berlin, Germany")))
map_widget.set_position(coordinates[0], coordinates[1])
map_widget.set_zoom(12)
# Run the Tkinter event loop

root.mainloop()
