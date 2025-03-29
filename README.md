# URBANALIZER #
An app to evaluate cities around the world and visualize information. The app is in its development phase and requires a lot of improvements. 

Attribution: 
- https://github.com/bahar/WorldCityLocations
- Nominatim
- OpenStreetMap
- CustomTkinter by Tom Schimansky
- Overpass Turbo

# Usage:
There are two main files: main_with_txt.py and main_manual_with_geojson.py. 
They are essentially the same app, except that they differ in their file reading type. 
For main_with_txt.py, there is a built in function to extract city data via the "Overpass Turbo" Tab.
For main_manual_with_geojson.py, one must extract city data on the browser Overpass Turbo API and manually has to rename the downloaded geojson file and put it in the directory: Data

This application was created for a project as part of a German high school graduation assignment in the subjects geography and computer science.
