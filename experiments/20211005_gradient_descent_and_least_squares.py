import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join('..', 'libs'))

from gradient_descent import gradient_descent_algorithm
from objective_functions import LeastSquares
from line_searches import BacktrackingLineSearch

#
# 目的関数      : 最小二乗法
# 最適化        : 勾配降下法 
# ステップサイズ : バックトラッキング直線探索
#

print("Hello")

np.random.seed(seed=0)

A = np.array([[2,3],[4,-5]])  # Ax = b の解析解は x1 = 2, x2 = 1
b = np.array([7,3])

least_squares_obj = LeastSquares(A, b)
blts_obj = BacktrackingLineSearch(least_squares_obj)
x_init = np.array([7, 1])

(x_arr, loss_arr, eta_arr) = gradient_descent_algorithm(least_squares_obj, blts_obj, x_init, 1e-4)

iteration_axis = np.arange(0, loss_arr.shape[0])
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(iteration_axis, loss_arr, marker='x', color = 'b')
ax[0, 1].plot(x_arr[:,0], x_arr[:,1], marker='x', color = 'b')
ax[1, 0].plot(iteration_axis, loss_arr - 0, marker='x', color = 'b')
# ax[1, 0].plot(iteration_axis, loss_arr - loss_arr[loss_arr.shape[0]-1], marker='x', color = 'b')
ax[1, 1].plot(iteration_axis, eta_arr, marker='x', color = 'b')

ax[0, 1].set_xlim(0,8)
ax[0, 1].set_ylim(0,5)

ax[1, 0].set_yscale('log')

fig.suptitle('Implementation of pure least squares')

ax[0, 0].set_xlabel('iteration')
ax[0, 0].set_ylabel('objective function value')
ax[0, 1].set_xlabel('x1')
ax[0, 1].set_ylabel('x2')
ax[1, 0].set_xlabel('iteration')
ax[1, 0].set_ylabel('deviation log scale')
ax[1, 1].set_xlabel('iteration')
ax[1, 1].set_ylabel('step size')

plt.show()

