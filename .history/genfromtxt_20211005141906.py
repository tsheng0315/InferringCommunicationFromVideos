from numpy import genfromtxt
data = genfromtxt(f'./videosLabelled/2/data.csv', delimiter='"', names=True)
print(data)