import json

# Class descriptions:
sign_dict = {
    1: "No entry",
    2: "No parking / waiting",
    3: "No turning",
    4: "Max Speed",
    5: "Other prohibition signs",
    6: "Warning",
    7: "Mandatory"
}

json_path = "data/za_traffic_2020/traffic_train/train_traffic_sign_dataset.json"
csv_path = "data/za_traffic_2020/traffic_train/train_traffic_sign_dataset.csv"
with open(csv_path, "w") as csv_file:
    with open(json_path) as json_file:
        data = json.load(json_file)
        annotations = data['annotations']
        for p in annotations:
            print('Bbox: ' + str(p['bbox']))
            print('Image: ' + str(p['image_id']))
            print('category_id: ' + str(sign_dict[p['category_id']]))
            csv_file.write(
                "images/{}.png,{},{},{},{},{}\n".format(p['image_id'], p['bbox'][0], p['bbox'][1], p['bbox'][0] + p['bbox'][2],
                                                        p['bbox'][1] + p['bbox'][3], sign_dict[p['category_id']]))
