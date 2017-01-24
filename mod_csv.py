import os
import csv

r = csv.reader(open('data/driving_log.csv'))
lines = [l for l in r]

for line in lines:
    for idx, el in enumerate(line):
        el = el.replace('/Users/Sergio/sdcn/git/CarND-Behavioral-Cloning-P3/data/', '')
        line[idx] = el

print(lines[len(lines)-1])

writer = csv.writer(open('data/driving_log.csv', 'w'))
writer.writerows(lines)
