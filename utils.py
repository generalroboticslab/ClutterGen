import json
from collections import OrderedDict
import os
import csv
import pprint



def read_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def write_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

def write_csv_line(result_file_path, result):
    """ write a line in a csv file; create the file and write the first line if the file does not already exist """
    # pp = pprint.PrettyPrinter(indent=4)
    # pp.pprint(result)
    result = OrderedDict(result)
    file_exists = os.path.exists(result_file_path)
    with open(result_file_path, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, result.keys())
        if not file_exists: writer.writeheader()
        writer.writerow(result)