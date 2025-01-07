import numpy as np

with open('heading_log.txt', 'r') as f:
   headings = np.array([float(x) for x in f.readlines()])
   
print(f"Mean: {np.mean(headings):.3f}")
print(f"Std: {np.std(headings):.3f}")
