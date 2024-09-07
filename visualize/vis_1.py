import numpy as np

layer_res = [30, 30, 30, 30, 30, 30, 30, 30, 800, 800, 800, 800, 800, 800, 1000, 70, 70, 1000]
layer_dim = [34, 34, 34, 34, 34, 34, 34, 34, 2, 2, 2, 2, 2, 2, 1]
layer_dim = [34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34, 34]
layer_dim = [27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27, 27]

paramsum = 0
for elt in layer_dim:
    paramsum += (elt * elt)

print(paramsum)
