import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join('..', 'libs'))

from newton_method import newton_method_algorithm
from quasi_newton_bfgs_method import bfgs_method_algorithm
from quasi_newton_dfp_method import dfp_method_algorithm
from quasi_newton_sr1_method import sr1_method_algorithm
from objective_functions_for_examples import NonQuadraticExample1
from line_searches import BacktrackingLineSearch

#
# 目的関数      : 非二次関数
# 最適化        : 準ニュートン法 SR1
# ステップサイズ : バックトラッキング直線探索
#
# Boyd and Vandenberghe, Convex Optimization Figure 9.19 の追試
# ニュートン法，準ニュートン法 BFGS, 準ニュートン法 DFP と比較した
#
# 講義スライド p21 左図 も同じ
# https://web.stanford.edu/class/ee364a/lectures/unconstrained.pdf
#

print("Hello")

nonquad_obj = NonQuadraticExample1()
ls_obj = BacktrackingLineSearch(nonquad_obj, 0.1, 0.7)  # パラメータは教科書通り

x_init = np.array([-1.0, 1.0])
tol = 1e-7
tol_quasi = 1e-4

(x_arr, loss_arr, eta_arr) = newton_method_algorithm(nonquad_obj, ls_obj, x_init, tol)
(x_arr_bfgs, loss_arr_bfgs, eta_arr_bfgs) = bfgs_method_algorithm(nonquad_obj, ls_obj, x_init, tol_quasi)
(x_arr_dfp, loss_arr_dfp, eta_arr_dfp) = dfp_method_algorithm(nonquad_obj, ls_obj, x_init, tol_quasi)
(x_arr_sr1, loss_arr_sr1, eta_arr_sr1) = sr1_method_algorithm(nonquad_obj, ls_obj, x_init, tol_quasi)

# print(loss_arr[loss_arr.shape[0]-1])
p = 2.5592666966582156   # tol = 1e-22 としたときの loss_arr_bfgs[loss_arr.shape[0]-1]

iteration_axis = np.arange(0, loss_arr.shape[0])
iteration_axis_bfgs = np.arange(0, loss_arr_bfgs.shape[0])
iteration_axis_dfp = np.arange(0, loss_arr_dfp.shape[0])
iteration_axis_sr1 = np.arange(0, loss_arr_sr1.shape[0])

fig, ax = plt.subplots(2, 2)

ax[0, 0].plot(iteration_axis, loss_arr, marker='x', color = 'b', linestyle = 'solid', label = 'Newton')
ax[0, 1].plot(x_arr[:,0], x_arr[:,1], marker='x', color = 'b', linestyle = 'solid', label = 'trajectory in x1-x2 plane')
ax[1, 0].plot(iteration_axis, loss_arr - p, marker='x', color = 'b', linestyle = 'solid', label = 'f(x^k) - f*')
ax[1, 1].plot(iteration_axis, eta_arr, marker='x', color = 'b', linestyle = 'solid', label = 'backtracking line search')

ax[0, 0].plot(iteration_axis_bfgs, loss_arr_bfgs, marker='x', color = 'm', linestyle = 'solid', label = 'Quasi-Newton BFGS')
ax[0, 1].plot(x_arr_bfgs[:,0], x_arr_bfgs[:,1], marker='x', color = 'm', linestyle = 'solid', label = 'trajectory in x1-x2 plane')
ax[1, 0].plot(iteration_axis_bfgs, loss_arr_bfgs - p, marker='x', color = 'm', linestyle = 'solid', label = 'f(x^k) - f*')
ax[1, 1].plot(iteration_axis_bfgs, eta_arr_bfgs, marker='x', color = 'm', linestyle = 'solid', label = 'backtracking line search')

ax[0, 0].plot(iteration_axis_dfp, loss_arr_dfp, marker='x', color = 'g', linestyle = 'dotted', label = 'Quasi-Newton DFP')
ax[0, 1].plot(x_arr_dfp[:,0], x_arr_dfp[:,1], marker='x', color = 'g', linestyle = 'dotted', label = 'trajectory in x1-x2 plane')
ax[1, 0].plot(iteration_axis_dfp, loss_arr_dfp - p, marker='x', color = 'g', linestyle = 'dotted', label = 'f(x^k) - f*')
ax[1, 1].plot(iteration_axis_dfp, eta_arr_dfp, marker='x', color = 'g', linestyle = 'dotted', label = 'backtracking line search')

ax[0, 0].plot(iteration_axis_sr1, loss_arr_sr1, marker='x', color = 'c', linestyle = 'dotted', label = 'Quasi-Newton SR1')
ax[0, 1].plot(x_arr_sr1[:,0], x_arr_sr1[:,1], marker='x', color = 'c', linestyle = 'dotted', label = 'trajectory in x1-x2 plane')
ax[1, 0].plot(iteration_axis_sr1, loss_arr_sr1 - p, marker='x', color = 'c', linestyle = 'dotted', label = 'f(x^k) - f*')
ax[1, 1].plot(iteration_axis_sr1, eta_arr_sr1, marker='x', color = 'c', linestyle = 'dotted', label = 'backtracking line search')

ax[0, 1].set_xlim(-1.5,1.5)
ax[0, 1].set_ylim(-1.5,1.5)

ax[1, 0].set_ylim(1e-15,1e5)
ax[1, 0].set_yscale('log')
ax[1, 0].set_yticks([1e-15,1e-10,1e-5,1,1e5])

fig.suptitle('Boyd, Convex Optimization, Figure 9.19 with Quasi-Newton result')

ax[0, 0].set_xlabel('iteration')
ax[0, 0].set_ylabel('objective function value')
ax[0, 1].set_xlabel('x1')
ax[0, 1].set_ylabel('x2')
ax[1, 0].set_xlabel('iteration')
ax[1, 0].set_ylabel('f(x^k) - f*')
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

