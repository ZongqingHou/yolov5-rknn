import json
import csv

id_dict = {}
with open('/home/hdd/volums/yolov5-rknn/tools/开口扳手.txt', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        if row[0] not in id_dict:
            id_dict[row[0]] = row[1]
        else:
            print(row[0])
            print("error")

colorring_collection = []
with open('/home/hdd/volums/yolov5-rknn/tools/tools.txt', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        colorring_collection.append({"name": row[0], "label": id_dict[row[0]], "subLabel": row[-1], "guige": "{},{}".format(row[0], row[1])})

result = {"data": colorring_collection}
en_json = json.dumps(result, ensure_ascii=False)

with open('output.json', 'w', encoding='utf-8') as f:
    f.write(en_json)
