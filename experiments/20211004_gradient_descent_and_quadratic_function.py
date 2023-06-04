import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join('..', 'libs'))

from gradient_descent import gradient_descent_algorithm
from objective_functions import QuadraticFunction
from line_searches import BacktrackingLineSearch

#
# 目的関数      : 二次関数
# 最適化        : 勾配降下法 
# ステップサイズ : バックトラッキング直線探索
#
# 条件数が大きいとジグザグに収束することを確認するテスト
# Boyd, Convex Optimization Fig 9.2 の追試 (Boyd は 厳密直線探索だがこのテストはバックトラッキング直線探索)
# 講義スライド 10.8 も同じ
# https://web.stanford.edu/class/ee364a/lectures/unconstrained.pdf
# 
# 最適値に対する偏差の対数が線形、つまり一次収束していることが確認できる
#


print("Hello")

Q = np.array([[1, 0], [0, 10]])

quad_obj = QuadraticFunction(Q, regularization = 'none')
ls_obj = BacktrackingLineSearch(quad_obj, 0.1, 0.7)

x_init = np.array([10,1])
(x_arr, loss_arr, eta_arr) = gradient_descent_algorithm(quad_obj, ls_obj, x_init, 1e-1)


iteration_axis = np.arange(0, loss_arr.shape[0])
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(iteration_axis, loss_arr, marker='x', color = 'b')
ax[0, 1].plot(x_arr[:,0], x_arr[:,1], marker='x', color = 'b')
ax[1, 0].plot(iteration_axis, loss_arr - 0, marker='x', color = 'b')
# ax[1, 0].plot(iteration_axis, loss_arr - loss_arr[loss_arr.shape[0]-1], marker='x', color = 'b')
ax[1, 1].plot(iteration_axis, eta_arr, marker='x', color = 'b')

ax[0, 1].set_xlim(-11,11)
ax[0, 1].set_ylim(-5,5)

ax[1, 0].set_yscale('log')

fig.suptitle('Implementation of Boyd,Convex Optimization,Fig 9.2')

ax[0, 0].set_xlabel('iteration')
ax[0, 0].set_ylabel('objective function value')
ax[0, 1].set_xlabel('x1')
ax[0, 1].set_ylabel('x2')
ax[1, 0].set_xlabel('iteration')
ax[1, 0].set_ylabel('deviation log scale')
ax[1, 1].set_xlabel('iteration')
ax[1, 1].set_ylabel('step size')

plt.show()

