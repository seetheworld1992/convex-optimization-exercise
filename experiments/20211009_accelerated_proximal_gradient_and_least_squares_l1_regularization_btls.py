import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join('..', 'libs'))

from proximal_gradient import proximal_gradient_algorithm
from accelerated_proximal_gradient import accelerated_proximal_gradient_algorithm
from objective_functions import LeastSquares
from line_searches import ConstantLineSearch
from line_searches import BacktrackingLineSearchForProximal
from proximal_operators import ProximalOperatorL1Norm

#
# 目的関数      : 最小二乗問題 + L1 正則化
# 最適化        : 近接勾配法
#                 近接勾配法 + バックトラッキング
#                 加速付き近接勾配法
#                 加速付き近接勾配法 + バックトラッキング
#
# Vandenberghe先生 講義スライド (右下のスライド番号)14 の追試
# http://www.seas.ucla.edu/~vandenbe/236C/lectures/fista.pdf
# 

print("Hello")

np.random.seed(seed=0)

row_dim = 200
column_dim = 100
A = np.random.randint(-20, 20, (row_dim, column_dim))
b = np.random.randint(-20, 20, (row_dim))
lamb = 1

least_squares_obj = LeastSquares(A, b, 'l1', lamb)
least_squares_btls_obj = LeastSquares(A, b, 'none', 0)
prox_1norm_obj = ProximalOperatorL1Norm(lamb)

eig_v, v = np.linalg.eig(np.dot(A.T, A))
eig_v_max = np.max(eig_v)
eta = 1/eig_v_max

constant_ls_obj = ConstantLineSearch(eta)
beta = 0.7
btls_obj = BacktrackingLineSearchForProximal(least_squares_btls_obj, prox_1norm_obj, beta)

x_init = np.zeros(column_dim)
iteration_end = 100

(x_arr, loss_arr, eta_arr) = proximal_gradient_algorithm(least_squares_obj, prox_1norm_obj, constant_ls_obj, x_init, iteration_end)
(x_arr_btls, loss_arr_btls, eta_arr_btls) = proximal_gradient_algorithm(least_squares_obj, prox_1norm_obj, btls_obj, x_init, iteration_end)
(x_arr_acc, loss_arr_acc, eta_arr_acc) = accelerated_proximal_gradient_algorithm(least_squares_obj, prox_1norm_obj, constant_ls_obj, x_init, iteration_end)
(x_arr_acc_btls, loss_arr_acc_btls, eta_arr_acc_btls) = accelerated_proximal_gradient_algorithm(least_squares_obj, prox_1norm_obj, btls_obj, x_init, iteration_end)

# print(loss_arr_acc[loss_arr_acc.shape[0]-1])
p = 6915.958802245224  # row_dim=200, column_dim=100 で iteration=10000 の目的関数値を最適値とみなす

# # 非ゼロ要素の数
# non_zero_cnt_arr = np.zeros(x_arr.shape[0])
# non_zero_cnt_arr_acc = np.zeros(x_arr.shape[0])
# for i in range(x_arr.shape[0]):
#     non_zero_cnt_arr[i] = np.sum(np.abs( x_arr[i] ) > 1e-16)
#     non_zero_cnt_arr_acc[i] = np.sum(np.abs( x_arr[i] ) > 1e-16)

iteration_axis = np.arange(0, loss_arr.shape[0])
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(iteration_axis, loss_arr, marker='None', color ='b', linestyle = 'dotted', label='PG')
# ax[0, 1].plot(iteration_axis, non_zero_cnt_arr, marker='None', color = 'b', linestyle = 'dotted', label='number of nonzero component')
ax[1, 0].plot(iteration_axis, (loss_arr - p) / p, marker='None', color ='b', linestyle = 'dotted', label='( f(x^k) - f* ) / f*')
ax[1, 1].plot(iteration_axis, eta_arr, marker='None', color = 'b', linestyle = 'dotted', label='constant step size')

ax[0, 0].plot(iteration_axis, loss_arr_btls, marker='None', color ='g', linestyle = 'dotted', label='PG + btls')
# ax[0, 1].plot(iteration_axis, non_zero_cnt_arr, marker='None', color = 'g', linestyle = 'dotted', label='number of nonzero component')
ax[1, 0].plot(iteration_axis, (loss_arr_btls - p) / p, marker='None', color ='g', linestyle = 'dotted', label='( f(x^k) - f* ) / f*')
ax[1, 1].plot(iteration_axis, eta_arr_btls, marker='None', color = 'g', linestyle = 'dotted', label='backtracking line search')

ax[0, 0].plot(iteration_axis, loss_arr_acc, marker='None', color ='m', linestyle = 'solid', label='accelerated PG')
# ax[0, 1].plot(iteration_axis, non_zero_cnt_arr_acc, marker='None', color = 'm', linestyle = 'solid', label='number of nonzero component')
ax[1, 0].plot(iteration_axis, (loss_arr_acc - p) / p, marker='None', color ='m', linestyle = 'solid', label='( f(x^k) - f* ) / f*')
ax[1, 1].plot(iteration_axis, eta_arr_acc, marker='None', color = 'm', linestyle = 'solid', label='constant step size')

ax[0, 0].plot(iteration_axis, loss_arr_acc_btls, marker='None', color ='c', linestyle = 'solid', label='accelerated PG + btls')
# ax[0, 1].plot(iteration_axis, non_zero_cnt_arr_acc, marker='None', color = 'c', linestyle = 'solid', label='number of nonzero component')
ax[1, 0].plot(iteration_axis, (loss_arr_acc_btls - p) / p, marker='None', color ='c', linestyle = 'solid', label='( f(x^k) - f* ) / f*')
ax[1, 1].plot(iteration_axis, eta_arr_acc_btls, marker='None', color = 'c', linestyle = 'solid', label='backtracking line search')


ax[1, 0].set_yscale('log')

fig.suptitle('Vandenberghe, EE 236C (Spring 2013-2014), 7. Fast proximal gradient methods, 1-norm regularized least-squares')

ax[0, 0].set_xlabel('iteration')
ax[0, 0].set_ylabel('objective function value')
# ax[0, 1].set_xlabel('iteration')
# ax[0, 1].set_ylabel('number of nonzero component')
ax[1, 0].set_xlabel('iteration')
ax[1, 0].set_ylabel('( f(x^k) - f* ) / f*')
ax[1, 1].set_xlabel('iteration')
ax[1, 1].set_ylabel('step size')

ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
# ax[1, 1].legend()

plt.show()

