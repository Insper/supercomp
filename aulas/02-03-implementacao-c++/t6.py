import numpy as np
import numpy.random
import numpy.linalg

n_points = [1, 5, 20, 120, 2300]

for i, n in enumerate(n_points):
    mat = np.random.rand(n, 2) * 10000
    with open(f't6-in-{i}.txt', 'w') as f:
        print(n, file=f)
        for p in mat:
            print(p[0], p[1], file=f)
    
    with open(f't6-out-{i}.txt', 'w') as f:
        for i, p in enumerate(mat):
            for j, p2 in enumerate(mat):
                print(f'{np.linalg.norm(p-p2):.2f}', end=' ', file=f)
            print('', file=f)
