import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join('..', 'libs'))

from subgradient_method import subgradient_algorithm
from objective_functions import LeastL1Norm
from step_size_rules import NonsummableDiminishing1
from step_size_rules import NonsummableDiminishing2

#
# 目的関数      : 最小L1ノルム
# 最適化        : 劣勾配法
# ステップサイズ : 非総和可能減少(2種類)
#
# Vandenberghe先生 講義スライド 3.9 の追試
# http://www.seas.ucla.edu/~vandenbe/236C/lectures/sgmethod.pdf
# 

print("Hello")

np.random.seed(seed=0)

row_dim = 250
column_dim = 50
A = np.random.randint(-20, 20, (row_dim, column_dim))
b = np.random.randint(-20, 20, (row_dim))
lamb = 0

least_l1_norm_obj = LeastL1Norm(A, b, 'none', lamb)

h = 0.01
step_size_rule_obj0 = NonsummableDiminishing1(h)
step_size_rule_obj1 = NonsummableDiminishing2(h)

x_init = np.zeros(column_dim)
iteration_end = 5000

(x_arr_0, loss_arr_0, eta_arr_0) = subgradient_algorithm(least_l1_norm_obj, step_size_rule_obj0, x_init, iteration_end)
(x_arr_1, loss_arr_1, eta_arr_1) = subgradient_algorithm(least_l1_norm_obj, step_size_rule_obj1, x_init, iteration_end)

# f_best の作成
empty_arr = np.empty(0)
f_best_arr_0 = np.append(empty_arr, loss_arr_0[0])
f_best_arr_1 = np.append(empty_arr, loss_arr_1[0])
for i in range(1, x_arr_1.shape[0]):    
    f_best_arr_0 = np.append(f_best_arr_0, np.min(loss_arr_0[0:i]))
    f_best_arr_1 = np.append(f_best_arr_1, np.min(loss_arr_1[0:i]))

print(f_best_arr_1[f_best_arr_1.shape[0] - 1])
p = 2148.3433246182562    # row_dim = 250, column_dim = 50 で iteration=30000 の f_best_arr_1 を最適値とみなす

iteration_axis = np.arange(0, loss_arr_0.shape[0])
fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(iteration_axis, loss_arr_0, marker='', linewidth = 0.5, color = 'b', label = 'objective funciton value')
ax[0, 1].plot(iteration_axis, (f_best_arr_0 - p) / p, marker='', linewidth = 1.0, color = 'b', label = '( f_best,k - f* ) / f*')
ax[1, 0].plot(iteration_axis, (loss_arr_0 - p) / p, marker='',  linewidth = 0.5, color = 'b', label = '( f(x^k) - f* ) / f*')
ax[1, 1].plot(iteration_axis, eta_arr_0, marker='', linewidth = 1.0, color = 'b', label = 't_k = 0.01 / sqrt(k + 1)')

ax[0, 0].plot(iteration_axis, loss_arr_1, marker='', linewidth = 0.5, color = 'm', label = 'objective funciton value')
ax[0, 1].plot(iteration_axis, (f_best_arr_1 - p) / p, marker='', linewidth = 1.0, color = 'm', label = '( f_best,k - f* ) / f*')
ax[1, 0].plot(iteration_axis, (loss_arr_1 - p) / p, marker='',  linewidth = 0.5, color = 'm', label = '( f(x^k) - f* ) / f*')
ax[1, 1].plot(iteration_axis, eta_arr_1, marker='', linewidth = 1.0, color = 'm', label = 't_k = 0.01 / (k + 1)')


ax[0, 0].set_xlim(0, iteration_end)
ax[0, 1].set_xlim(0, iteration_end)
ax[1, 0].set_xlim(0, iteration_end)
ax[1, 1].set_xlim(0, iteration_end)


ax[0, 1].set_yscale('log')
ax[0, 1].set_ylim(1e-5, 10)
ax[1, 0].set_yscale('log')
ax[1, 0].set_ylim(1e-5, 10)


fig.suptitle('Vandenberghe, ECE236C (Spring 2020), 3. Subgradient method, Example: 1-norm minimization')

ax[0, 0].set_xlabel('iteration')
ax[0, 0].set_ylabel('objective function value')
ax[0, 1].set_xlabel('iteration')
ax[0, 1].set_ylabel('( f_best,k - f* ) / f*')
ax[1, 0].set_xlabel('iteration')
ax[1, 0].set_ylabel('( f(x^k) - f* ) / f*')
ax[1, 1].set_xlabel('iteration')
ax[1, 1].set_ylabel('step size')


ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()

ax[0, 0].grid()
ax[0, 1].grid()
ax[1, 0].grid()
ax[1, 1].grid()

plt.show()

