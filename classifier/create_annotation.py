import numpy as np 
import csv
import pandas as pd 
import os

file = open('annotation_val.csv', 'w')
writer = csv.writer(file)
path = 'val/'
for cl in range(0, 7):
    for img_path in os.listdir(path + str(cl)):
        row = [str(cl) + '/' + img_path, cl]
        writer.writerow(row)
