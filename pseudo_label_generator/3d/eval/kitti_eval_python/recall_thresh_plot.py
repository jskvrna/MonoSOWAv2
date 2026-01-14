import matplotlib.pyplot as plt
import numpy as np

# read data from file
with open('/path/to/res_thresh_regnety.txt', 'r') as f:  # (redacted)
    lines = f.readlines()
    data = [line.strip().split() for line in lines]

x = np.arange(100)
x = x/100.
# extract x and y values from data
y0 = [float(row[0]) for row in data]
y1 = [float(row[1]) for row in data]
y2 = [float(row[2]) for row in data]
y3 = [float(row[3]) for row in data]
y4 = [float(row[4]) for row in data]
y5 = [float(row[5]) for row in data]

y0 = np.array(y0)
y1 = np.array(y1)
y2 = np.array(y2)
y3 = np.array(y3)
y4 = np.array(y4)
y5 = np.array(y5)


rec_easy = y0/(y0+y1)
rec_mod = y2/(y2+y3)
rec_hard = y4/(y4+y5)
# plot data
fig, ax = plt.subplots()
ax.plot(x, rec_easy, label='Easy')
ax.plot(x, rec_mod, label='Moderate')
ax.plot(x, rec_hard, label='Hard')
ax.legend()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Recall based on threshold')
plt.savefig("/path/to/recall_curve.png")  # (redacted)
