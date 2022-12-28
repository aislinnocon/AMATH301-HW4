
import numpy as np
import scipy.linalg


# Problem 1
q = -np.ones(113)
Q = np.diag(q, -1)
Z = np.diag(q, 1)
u = 2*np.ones(114)
U = np.diag(u)
A114 = U + Q + Z
A1 = A114.copy()


rho = np.zeros(114)
for j in range(114):
    rho[j] = 2*(1 - np.cos((53*np.pi)/115)) * np.sin((53*np.pi*(j+1))/115)

rho = rho.reshape(114,1)
A2 = rho.copy()

# Problem 2
A = A1.copy()
b = A2.copy()
P = np.diag(np.diag(A))
T = A - P

M = -scipy.linalg.solve(P, T)
w, V = np.linalg.eig(M)

A3 = np.max(np.abs(w))
print("A3 = ", A3)

tolerance = 1e-5
x0 = np.ones((114, 1))
X = np.zeros((114, 12000))
X[:, 0:1] = x0

for k in range(12000):
    X[:, (k+1):(k+2)] = scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + b, lower=True)
    if np.max(np.abs(X[:, k+1] - X[:, k])) < tolerance:
        break

# print(X)
# print(X.shape)
A4 = X[:, (k+1):(k+2)]
#print(A4)
# print(A4.shape)
A5 = k + 2
#print(A5)


ans = np.zeros((114,1))
for j in range(114):
    ans[j] = np.sin((53*np.pi*(j+1))/115)

A6 = np.max(np.abs(A4 - ans))
print("A6 = ", A6)

# Problem 3
P = np.tril(A1.copy())
T = A1.copy() - P
b = A2.copy()

M = -scipy.linalg.solve(P, T)
w, V = np.linalg.eig(M)
A7 = np.max(np.abs(w))
# print("A7 = ", A7)

tolerance = 1e-5
x0 = np.ones((114, 1))
X = np.zeros((114, 7001))
X[:, 0:1] = x0

for k in range(7000):
    X[:, (k+1):(k+2)] = scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + b, lower=True)
    if np.max(np.abs(X[:, k+1] - X[:, k])) < tolerance:
        break
X = X[:, :(k+2)]
# print(X)
A8 = X[:, (k+1):(k+2)]
# print(A8)
# print(A8.shape)
A9 = k + 2
# print(A9)

A10 = np.max(np.abs(A8 - ans))
# print(A10)

# Problem 4
D = np.diag(2*np.ones(114)) + np.zeros((114,114))
L = np.diag(-np.ones(113), -1) + np.zeros((114,114))
P = (1/1.5)*D + L
A11 = P.copy()
# print(A11)

U = np.diag(-np.ones(113), 1) + np.zeros((114,114))
T = ((1.5 - 1)/1.5)*D + U
A = D + U + L

M = -scipy.linalg.solve(P, T)
w, V = np.linalg.eig(M)

A12 = np.max(np.abs(w))

tolerance = 1e-5
x0 = np.ones((114, 1))
X = np.zeros((114, 3001))
X[:, 0:1] = x0

for k in range(3000):
    X[:, (k+1):(k+2)] = scipy.linalg.solve_triangular(P, -T @ X[:, k:(k+1)] + b, lower=True)
    if np.max(np.abs(X[:, k+1] - X[:, k])) < tolerance:
        break
X = X[:, :(k+2)]
A13 = X[:, (k+1):(k+2)]
A14 = k + 2

A15 = np.max(np.abs(A13 - ans))
