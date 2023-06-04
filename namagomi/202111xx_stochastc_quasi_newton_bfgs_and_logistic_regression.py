import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join('..', 'optimization_algorithm'))
# sys.path.append(os.path.join('..', 'data_sets'))

from objective_functions import LogisticLoss
from line_searches import ConstantLineSearch
from step_size_rules import ConstantStepSize
from step_size_rules import StepSizeRuleForObfgs
from gradient_descent import gradient_descent_algorithm
from stochastic_gradient_descent import stochastic_gradient_descent_algorithm
from stochastic_gradient_descent_mini_batch import mini_batch_sgd_algorithm
from stochastic_variance_reduced_gradient import svrg_algorithm
from stochastic_quasi_newton_bfgs import stochastic_bfgs_method_algorithm
from stochastic_quasi_newton_l_bfgs import stochastic_l_bfgs_method_algorithm


# 目的関数      : ロジスティックロス関数
# 最適化        : GD, SGD, Mini-Batch SGD (batch = 10,100), SVRG 
# ステップサイズ : 全て固定
# データセット  : https://www.statsmodels.org/dev/datasets/generated/fair.html
#                特徴量を二つ (occupation, occupation_husb) 削除して.csv を作成
# 

cwd  = os.getcwd()
print('getcwd:      ', os.getcwd())
print('__file__:    ', __file__)

# data = np.loadtxt('../data_sets/data_set_affairs_remove_occupation.csv', delimiter=',',skiprows = 1)
data = np.loadtxt('data_set_affairs_remove_occupation.csv', delimiter=',',skiprows = 1)
data_matrix = data[:,0:6]
label_vec = data[:,6]
label_vec[0 < label_vec] = 1

# fix random number seed
np.random.seed(seed=0)

lamb = 1
logistic_obj = LogisticLoss(data_matrix, label_vec, 'l2', lamb)

# for full_gradient
step_size_gd = 0.002
constant_ls_obj = ConstantLineSearch(step_size_gd)
tol = 1e-2

# for stochastic
h_sgd =    0.0001
h_mb_10 =  0.0006
h_mb_100 = 0.001
h_svrg =   0.002
step_size_rule_obj_sgd = ConstantStepSize(h_sgd)
step_size_rule_obj_mb_10 = ConstantStepSize(h_mb_10)
step_size_rule_obj_mb_100 = ConstantStepSize(h_mb_100)
step_size_rule_obj_svrg = ConstantStepSize(h_svrg)
x_init = np.zeros(data_matrix.shape[1])
batch_size_10 = 10
batch_size_100 = 100
outer_loop_freq = 20
iteration_end = 1000

(x_arr_gd, loss_arr_gd, eta_arr_gd) = gradient_descent_algorithm(logistic_obj, constant_ls_obj, x_init, tol)
# (x_arr_sgd, loss_arr_sgd, eta_arr_sgd) = stochastic_gradient_descent_algorithm(logistic_obj, step_size_rule_obj_sgd, x_init, iteration_end)
# (x_arr_sgd_mb_10, loss_arr_sgd_mb_10, eta_arr_sgd_mb_10) = mini_batch_sgd_algorithm(logistic_obj, step_size_rule_obj_mb_10, x_init, batch_size_10, iteration_end)
# (x_arr_sgd_mb_100, loss_arr_sgd_mb_100, eta_arr_sgd_mb_100) = mini_batch_sgd_algorithm(logistic_obj, step_size_rule_obj_mb_100, x_init, batch_size_100, iteration_end)
# (x_arr_svrg, loss_arr_svrg, eta_arr_svrg) = svrg_algorithm(logistic_obj, step_size_rule_obj_svrg, x_init, outer_loop_freq, iteration_end)

c = 0.1
lambd = 1
epsilon = 1e-10

batch_size_obfgs = 10
batch_size_olbfgs = 100
m = 5

eta_0_obfgs = 0.01
eta_0_olbfgs = 0.01
tau_obfgs = 1e5
tau_olbfgs = 1e5
step_size_rule_obj_obfgs = StepSizeRuleForObfgs(eta_0_obfgs, tau_obfgs)
step_size_rule_obj_olbfgs = StepSizeRuleForObfgs(eta_0_olbfgs, tau_olbfgs)
(x_arr_obfgs, loss_arr_obfgs, eta_arr_obfgs) = stochastic_bfgs_method_algorithm(logistic_obj, step_size_rule_obj_obfgs, x_init, batch_size_obfgs, c, lambd, epsilon, iteration_end)
(x_arr_olbfgs, loss_arr_olbfgs, eta_arr_olbfgs) = stochastic_l_bfgs_method_algorithm(logistic_obj, step_size_rule_obj_olbfgs, x_init, batch_size_olbfgs, c, lambd, epsilon, m, iteration_end)

# print(loss_arr_gd[loss_arr_gd.shape[0]-1])
p = 0.5991920355883356  # lamb = 1, l2, step_size_gd = 0.002 で tol = 1e-8 の目的関数値を最適値とみなす

iteration_axis_gd = np.arange(0, loss_arr_gd.shape[0])
# iteration_axis_sgd = np.arange(0, loss_arr_sgd.shape[0])
# iteration_axis_sgd_mb_10 = np.arange(0, loss_arr_sgd_mb_10.shape[0])
# iteration_axis_sgd_mb_100 = np.arange(0, loss_arr_sgd_mb_100.shape[0])
# iteration_axis_svrg = np.arange(0, loss_arr_svrg.shape[0])
iteration_axis_obfgs = np.arange(0, loss_arr_obfgs.shape[0])
iteration_axis_olbfgs = np.arange(0, loss_arr_olbfgs.shape[0])

fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(iteration_axis_gd, loss_arr_gd, marker='', linewidth = 0.7, color ='k', label='objective function value')
ax[0, 1].plot(iteration_axis_gd, np.zeros(iteration_axis_gd.shape[0]), marker='', linewidth = 1.0, color ='k', label='gradient descent')
ax[1, 0].plot(iteration_axis_gd, loss_arr_gd - p, marker='', linewidth = 0.7, color ='k', label='f(x^k) - f*')
ax[1, 1].plot(iteration_axis_gd, eta_arr_gd, marker= '', linewidth = 1.0, color = 'k', label='constant step size')

# ax[0, 0].plot(iteration_axis_sgd, loss_arr_sgd, marker='', linewidth = 0.7, color ='m', label='objective function value')
# ax[0, 1].plot(iteration_axis_sgd, np.zeros(iteration_axis_sgd.shape[0]), marker='', linewidth = 1.0, color ='m', label='SGD')
# ax[1, 0].plot(iteration_axis_sgd, loss_arr_sgd - p, marker='', linewidth = 0.7, color ='m', label='f(x^k) - f*')
# ax[1, 1].plot(iteration_axis_sgd, eta_arr_sgd, marker='', linewidth = 1.0, color = 'm', label='constant step size')

# ax[0, 0].plot(iteration_axis_sgd_mb_10, loss_arr_sgd_mb_10, marker='', linewidth = 0.7, color ='g', label='objective function value')
# ax[0, 1].plot(iteration_axis_sgd_mb_10, np.zeros(iteration_axis_sgd_mb_10.shape[0]), marker='', linewidth = 1.0, color ='g', label='Mini-Batch SGD 10')
# ax[1, 0].plot(iteration_axis_sgd_mb_10, loss_arr_sgd_mb_10 - p, marker='', linewidth = 0.7, color ='g', label='f(x^k) - f*')
# ax[1, 1].plot(iteration_axis_sgd_mb_10, eta_arr_sgd_mb_10, marker='', linewidth = 1.0, color = 'g', label='constant step size')

# ax[0, 0].plot(iteration_axis_sgd_mb_100, loss_arr_sgd_mb_100, marker='', linewidth = 0.7, color ='b', label='objective function value')
# ax[0, 1].plot(iteration_axis_sgd_mb_100, np.zeros(iteration_axis_sgd_mb_100.shape[0]), marker='', linewidth = 1.0, color ='b', label='Mini-Batch SGD 100')
# ax[1, 0].plot(iteration_axis_sgd_mb_100, loss_arr_sgd_mb_100 - p, marker='', linewidth = 0.7, color ='b', label='f(x^k) - f*')
# ax[1, 1].plot(iteration_axis_sgd_mb_100, eta_arr_sgd_mb_100, marker='', color = 'b', linewidth = 1.0, label='constant step size')

# ax[0, 0].plot(iteration_axis_svrg, loss_arr_svrg, marker='', linewidth = 0.7, color ='c', label='objective function value')
# ax[0, 1].plot(iteration_axis_svrg, np.zeros(iteration_axis_svrg.shape[0]), marker='', linewidth = 1.0, color ='c', label='SVRG')
# ax[1, 0].plot(iteration_axis_svrg, loss_arr_svrg - p, marker='', linewidth = 0.7, color ='c', label='f(x^k) - f*')
# ax[1, 1].plot(iteration_axis_svrg, eta_arr_svrg, marker='', linewidth = 1.0, color = 'c', label='constant step size')

ax[0, 0].plot(iteration_axis_obfgs, loss_arr_obfgs, marker='', linewidth = 0.7, color ='c', label='objective function value')
ax[0, 1].plot(iteration_axis_obfgs, np.zeros(iteration_axis_obfgs.shape[0]), marker='', linewidth = 1.0, color ='c', label='oBFGS')
ax[1, 0].plot(iteration_axis_obfgs, loss_arr_obfgs - p, marker='', linewidth = 0.7, color ='c', label='f(x^k) - f*')
ax[1, 1].plot(iteration_axis_obfgs, eta_arr_obfgs, marker='', linewidth = 1.0, color = 'c', label='constant step size')

ax[0, 0].plot(iteration_axis_olbfgs, loss_arr_olbfgs, marker='', linewidth = 0.7, color ='y', label='objective function value')
ax[0, 1].plot(iteration_axis_olbfgs, np.zeros(iteration_axis_olbfgs.shape[0]), marker='', linewidth = 1.0, color ='y', label='oLBFGS')
ax[1, 0].plot(iteration_axis_olbfgs, loss_arr_olbfgs - p, marker='', linewidth = 0.7, color ='y', label='f(x^k) - f*')
ax[1, 1].plot(iteration_axis_olbfgs, eta_arr_olbfgs, marker='', linewidth = 1.0, color = 'y', label='constant step size')

ax[0, 1].set_ylim(1,2)
ax[0, 1].tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False)
ax[1, 0].set_yscale('log')



fig.suptitle('Logistic Minimization with L2 Regularization')

ax[0, 0].set_xlabel('iteration')
ax[0, 0].set_ylabel('objective function value')
ax[1, 0].set_xlabel('iteration')
ax[1, 0].set_ylabel('f(x^k) - f*')
ax[1, 1].set_xlabel('iteration')
ax[1, 1].set_ylabel('step size')

# ax[0, 0].legend()
ax[0, 1].legend()
ax[1, 0].legend()
ax[1, 1].legend()

plt.show()

print('end')