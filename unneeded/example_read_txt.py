def read_txt_and_get_points(file: str) -> [(float, float)]:
    f = open(file, "r")
    raw = f.read()
    temp_list = raw.split(",")
    splitted_list = []
    for node in temp_list:
        splitted = node.split(" ")
        splitted_list.append(splitted)
    coords = []
    for node in splitted_list:
        if ">" in node[-1][4:][:-1]:
            coords.append((float(node[-2][4:][:-1]), float(node[-1][4:][:-2])))
        else:
            coords.append((float(node[-2][4:][:-1]), float(node[-1][4:][:-1])))
    return coords

print(read_txt_and_get_points("../OverpassResults/Berlin_Germany_TrainStations.txt"))