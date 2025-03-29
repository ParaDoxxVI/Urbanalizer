import overpy
import os
api = overpy.Overpass()

import requests

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
    ways = result.get_ways()
    relations = result.get_relations()
    with open(f'OverpassResults/{entry[0]}_{entry[1]}_{occasion}.txt', "w") as file:
        file.write(str(nodes+ways+relations))

    nodes = 0
    ways = 0
    relations = 0
    return




#city_objects
#transportation
#offices

#File:




print(find_overpass_area_id("Berlin", "Germany"))
