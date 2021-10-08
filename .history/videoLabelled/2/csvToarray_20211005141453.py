
data = genfromtxt(f'./videosLabelled/2/data.csv', delimiter=',', names=True)
print(data)
from csv import reader
# open file in read mode
with open('gaze_data_new.csv', 'r') as read_obj:
    # pass the file object to reader() to get the reader object
    csv_reader = reader(read_obj)
    # Get all rows of csv from csv_reader object as list of tuples
    list_of_tuples = list(map(tuple, csv_reader))
    # # display all rows of csv
    # print(list_of_tuples)