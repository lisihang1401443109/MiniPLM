import cvxpy as cp
import numpy as np
import matplotlib.pyplot as plt

dim = 1024

x = cp.Variable(dim)
y = np.random.rand(dim) * 2 - 1

objective = cp.Minimize(cp.sum_squares(x - y))
constraints = [cp.sum(x) == 1, x >= 0]
prob = cp.Problem(objective, constraints)

result = prob.solve()
print(x.value)

plt.plot(x.value)
plt.savefig('test_cvx.png')
plt.close()