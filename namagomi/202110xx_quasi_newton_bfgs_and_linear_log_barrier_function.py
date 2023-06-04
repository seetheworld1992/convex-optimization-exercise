import numpy as np
import matplotlib.pyplot as plt
import sys, os
# sys.path.append(os.path.join('..', 'optimization_algorithm'))
# sys.path.append(os.path.join('..', 'examples'))

from quasi_newton_bfgs_method import bfgs_method_algorithm
from newton_method import newton_method_algorithm
from objective_functions_for_examples import LinearPlusLogBarrier
from line_searches import BacktrackingLineSearch
from line_searches import ConstantLineSearch

# 目的関数      : 線形 + 対数バリア 関数
# 最適化        : 準ニュートン法 BFGS
# ステップサイズ : バックトラッキング直線探索

# Boyd and Vandenberghe, Convex Optimization Figure 9.19 の追試
# 講義スライド p21 左図 も同じ
# https://web.stanford.edu/class/ee364a/lectures/unconstrained.pdf

print("Hello")

np.random.seed(seed=0)

n = 100
m = 500
A = np.random.randint(-5, 5, (m, n))
b = np.random.randint(5, 12, (m))
c = 0.5 * np.ones(n)
# n = 10
# m = 50
# A = np.arange(0,m*n).reshape((m,n))
# b = np.ones(m)
# c = 0.5 * np.random.randint(0, 5, (n))

nonquad_obj = LinearPlusLogBarrier(A, b, c)
# constant_ls_obj = ConstantLineSearch(0.3)
ls_obj = BacktrackingLineSearch(nonquad_obj, 0.1, 0.7)  # パラメータは教科書通り

x_init = 10 * np.ones(n)
H_init = 0.001 * np.identity(n)
tol = 1e-6
(x_arr, loss_arr, eta_arr) = newton_method_algorithm(nonquad_obj, ls_obj, x_init, tol)
#(x_arr, loss_arr, eta_arr) = bfgs_method_algorithm(nonquad_obj, ls_obj, x_init, H_init, tol)

print(loss_arr[loss_arr.shape[0]-1])
p = -3756.545880025268   # constant_ls_obj で tol = 1e-12 としたときの loss_arr[loss_arr.shape[0]-1]

iteration_axis = np.arange(0, loss_arr.shape[0])
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(iteration_axis, loss_arr, marker='', color = 'b', label = 'objective funciton value')
# ax[0, 1].plot(x_arr[:,0], x_arr[:,1], marker='x', color = 'b', label = 'trajectory')
ax[1, 0].plot(iteration_axis, loss_arr - p, marker='x', color = 'b', label = 'f(x^k) - f*')
ax[1, 1].plot(iteration_axis, eta_arr, marker='x', color = 'b', label = 'backtracking line search')

# ax[0, 1].set_xlim(-1.5,1.5)
# ax[0, 1].set_ylim(-1.5,1.5)


ax[1, 0].set_yscale('log')
ax[1, 0].set_ylim(1e-12,1e3)
# ax[1, 0].set_yticks([1e-15,1e-10,1e-5,1,1e5])

fig.suptitle('Vandenberghe, ECE236C (Spring 2020), 15, Example (Linear + Log Barrier) ')

ax[0, 0].set_xlabel('iteration')
ax[0, 0].set_ylabel('objective function value')
# ax[0, 1].set_xlabel('x1')
# ax[0, 1].set_ylabel('x2')
ax[1, 0].set_xlabel('iteration')
ax[1, 0].set_ylabel('f(x^k) - f*')
ax[1, 1].set_xlabel('iteration')
ax[1, 1].set_ylabel('step size')

ax[0, 0].legend()
#ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()

ax[0, 0].grid()
#ax[0, 1].grid()
ax[1, 0].grid()
ax[1, 1].grid()

plt.show()

