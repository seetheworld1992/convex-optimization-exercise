import numpy as np
import matplotlib.pyplot as plt
import sys, os

sys.path.append(os.path.join('..', 'libs'))

from gradient_descent import gradient_descent_algorithm
from accelerated_proximal_gradient import accelerated_proximal_gradient_algorithm
from objective_functions import LogSumExpFunction
from line_searches import ConstantLineSearch
from proximal_operators import ProximalOperatorZero

#
# 目的関数      : Log-Sum-Exp
# 最適化        : 勾配降下法 vs 加速付き近接勾配法
# ステップサイズ : 固定ステップサイズ
#
# Vandenberghe先生 講義スライド 7.9 の追試
# 計算が重いので，係数行列は 2000×1000 ではなく大幅に妥協して 50×25 とした
# なんとなく再現できているっぽい
#
# https://www.seas.ucla.edu/~vandenbe/236C/lectures/fgrad.pdf
# 

print("Hello")

np.random.seed(seed=0)

row_dim = 50
column_dim = 25
A = np.random.randint(-20, 20, (row_dim, column_dim))
b = np.random.randint(-20, 20, (row_dim))

log_sum_exp_obj = LogSumExpFunction(A, b)

eta = 0.0020
constant_ls_obj = ConstantLineSearch(eta)
prox_zero_obj = ProximalOperatorZero()

x_init = np.zeros(column_dim)
tol = 1.6
iteration_end = 200

(x_arr, loss_arr, eta_arr) = gradient_descent_algorithm(log_sum_exp_obj, constant_ls_obj, x_init, tol)
(x_arr_acc, loss_arr_acc, eta_arr_acc) = accelerated_proximal_gradient_algorithm(log_sum_exp_obj, prox_zero_obj, constant_ls_obj, x_init, iteration_end)

# print(loss_arr_acc[loss_arr_acc.shape[0]-1])
p = 5.2715315081792875  # row_dim=50, column_dim=25 で iteration=1000 の目的関数値を最適値とみなす

iteration_axis = np.arange(0, iteration_end)
# iteration_axis = np.arange(0, loss_arr.shape[0])
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(iteration_axis, loss_arr[0:iteration_end], marker='x', color ='b', label='gradient descent')
# ax[0, 1].plot(iteration_axis, non_zero_cnt_arr, marker='x', color = 'b', label='number of nonzero component')
ax[1, 0].plot(iteration_axis, (loss_arr[0:iteration_end] - p) / np.abs(p), marker='x', color ='b', label='( f(x^k) - f* ) / f*')
ax[1, 1].plot(iteration_axis, eta_arr[0:iteration_end], marker='x', color = 'b', label='step size')

ax[0, 0].plot(iteration_axis, loss_arr_acc[0:iteration_end], marker='x', color ='m', label='accelerated proximal gradient')
# ax[0, 1].plot(iteration_axis, non_zero_cnt_arr_acc, marker='x', color = 'm', label='number of nonzero component')
ax[1, 0].plot(iteration_axis, (loss_arr_acc[0:iteration_end] - p) / np.abs(p), marker='x', color ='m', label='( f(x^k) - f* ) / f*')
ax[1, 1].plot(iteration_axis, eta_arr_acc[0:iteration_end], marker='x', color = 'm', label='step size')

# ax[0, 1].set_xlim(-2,0)
ax[1, 0].set_ylim(1e-6,10)

ax[1, 0].set_yscale('log')

fig.suptitle('Vandenberghe, ECE236C (Spring 2020), 7, Example: log-sum-exp')

ax[0, 0].set_xlabel('iteration')
ax[0, 0].set_ylabel('objective function value')
ax[0, 1].set_xlabel('iteration')
ax[0, 1].set_ylabel('number of nonzero component')
ax[1, 0].set_xlabel('iteration')
ax[1, 0].set_ylabel('( f(x^k) - f* ) / f*')
ax[1, 1].set_xlabel('iteration')
ax[1, 1].set_ylabel('step size')

ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()

plt.show()

