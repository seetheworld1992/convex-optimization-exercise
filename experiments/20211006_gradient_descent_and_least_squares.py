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
# 列フルランクであれば解析解が存在するので，勾配降下法を使う意味はない
# 勉強のため，一次収束収束していることを確認する 
# 

print("Hello")

np.random.seed(seed=0)

row_dim = 200
column_dim = 100
A = np.random.randint(-20, 20, (row_dim, column_dim))
b = np.random.randint(-20, 20, (row_dim))

least_squares_obj = LeastSquares(A, b)
btls_obj = BacktrackingLineSearch(least_squares_obj, 0.1, 0.7)  # パラメータは教科書通り

x_init = np.zeros(column_dim)

(x_arr, loss_arr, eta_arr) = gradient_descent_algorithm(least_squares_obj, btls_obj, x_init, 5)


x_analytic = np.dot( np.dot( np.linalg.inv( np.dot( A.T, A ) ) , A.T ) , b )  # 解析解
tmp = np.dot(A, x_analytic) - b
objective_analytic = np.dot(tmp, tmp) / 2  # 解析解による目的関数

iteration_axis = np.arange(0, loss_arr.shape[0])
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(iteration_axis, loss_arr, marker='x', color='b', label='objective function value')
# ax[0, 1].plot(x_arr[:,0], x_arr[:,1], marker='x', color = 'b')
# ax[1, 0].plot(iteration_axis, loss_arr - loss_arr[loss_arr.shape[0]-1], marker='x', color = 'b')
ax[1, 0].plot(iteration_axis, loss_arr - objective_analytic, marker='x', color='b', label='convergence error')
ax[1, 1].plot(iteration_axis, eta_arr, marker='x', color='b', label='step size')

# ax[0, 1].set_xlim(-2,0)
# ax[0, 1].set_ylim(-2,2)

ax[1, 0].set_yscale('log')

fig.suptitle('Implementation of gradient descent and least-squares')

ax[0, 0].set_xlabel('iteration')
ax[0, 0].set_ylabel('objective function value')
# ax[0, 1].set_xlabel('x1')
# ax[0, 1].set_ylabel('x2')
ax[1, 0].set_xlabel('iteration')
ax[1, 0].set_ylabel('convergence error')
ax[1, 1].set_xlabel('iteration')
ax[1, 1].set_ylabel('step size')


ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()

plt.show()

