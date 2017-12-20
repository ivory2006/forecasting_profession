import json
import csv
from pprint import pprint
with open('newDataSample.csv','r') as csvfile:
    reader = csv.DictReader(csvfile)
    handle = [row['Handle'] for row in reader]
    profession=[row['Profession'] for row in reader]
print(handle[0])
