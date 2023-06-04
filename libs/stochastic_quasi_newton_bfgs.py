import numpy as np
from numpy.core.numeric import identity
from numpy.lib.function_base import iterable
from extract_data_uniform import extract_row_according_to_randint_index

def update_bfgs_online(H, s, y, c):
    rho = 1 / np.dot(y,s)
    vec_Hy = np.dot(H, y)
    H_new = H - rho * np.outer(vec_Hy, s) - rho * np.outer( s, np.dot(y, H) ) + rho * rho * np.outer( s , np.dot(y, vec_Hy) * s) + c * rho * np.outer(s, s)
    
    return H_new


def stochastic_bfgs_method_algorithm(objective_obj, step_size_rule_obj, x_init, batch_size = 20, c = 0.1, lambd = 1, epsilon = 1e-10, iter_end = 6000):
    
    """
    確率的準ニュートン法 BFGS のアルゴリズム
    Nicol N. Schraudolph, Jin Yu, Simon G¨unter の
    A Stochastic Quasi-Newton Method for Online Convex Optimization のアルゴリズム oBFGS を実装
    http://proceedings.mlr.press/v2/schraudolph07a/schraudolph07a.pdf
　　
    目的関数が経験損失のようなsummationである場合を想定
    ラインサーチ機能は無し

    Parameters
    ----------
    objective_obj : class[ObjectiveFunction の具象クラス]
        目的関数の抽象クラスの具象クラスのインスタンス
    step_size_rule_obj : class[LineSearch の具象クラス]
        ステップサイズルールの抽象クラスの具象クラスのインスタンス
    x_init : ndarray[float64]
        初期値
        要素数は objective_obj の feat_matrix の axis 1 と同じ
    batch_size : int
        勾配計算に用いるバッチ数
    c : float64
        セカント方程式 s_k = x_{k+1} - x_k の係数 (0 < c <= 1)
    lambd : float64
        セカント方程式 s_k = grad_f_{k+1} - grad_f_k + lambd s_k の係数 (0 <= lambd)
    epsilon : float64
        ヘッセ行列の初期値の係数 (epsilon > 0)
    iter_end : int
        反復回数

    Returns
    -------
    x_arr : ndarray[float64, float64]
        点 x の計算結果
        axis 0: 反復回数
        axis 1: データ属性
    objective_arr : ndarray[float64]
        目的関数 objective_obj.objective_func(x) の計算結果
        axis 0: 反復回数
    eta_arr : ndarray[float64]
        ステップサイズ eta の計算結果
        axis 0: 反復回数
                    
    """

    feat_matrix = objective_obj.get_feat_matrix()
    label_vec = objective_obj.get_label_vec()

    # iteration = 0    
    objective_init = objective_obj.objective_func(x_init)
    (extracted_matrix, extracted_vec, extracted_int) = extract_row_according_to_randint_index(objective_obj.get_feat_matrix(), objective_obj.get_label_vec(), batch_size)
    grad_f = objective_obj.mini_batch_grad(x_init, extracted_matrix, extracted_vec)
    minus_bfgs_step = epsilon * grad_f
    iteration = 0
    eta = step_size_rule_obj.step_size_rule(minus_bfgs_step, iteration)
    
    # store    
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    objective_arr = np.append(empty_arr, objective_init)
    eta_arr = np.append(empty_arr, eta)

    # iteration = 1,2,...
    iteration += 1
    x = x_init
    first_time = True
    while iteration <= iter_end:

        # current iteration
        x_new = x - eta * minus_bfgs_step
        objective_new = objective_obj.objective_func(x_new)

        # for next iteration        
        grad_f_new = objective_obj.mini_batch_grad(x_new, extracted_matrix, extracted_vec)        
        s = - eta * minus_bfgs_step / c  # =  ( x_new - x ) / c
        y = grad_f_new - grad_f + lambd * s
        if first_time:
            Hessian_inv = np.dot(y,s) / np.dot(y,y) * np.identity(x_init.shape[0])
            first_time = False
        Hessian_inv = update_bfgs_online(Hessian_inv, s, y, c)
        x = x_new
        minus_bfgs_step = np.dot(Hessian_inv, grad_f_new)
        eta = step_size_rule_obj.step_size_rule(minus_bfgs_step, iteration)
        (extracted_matrix, extracted_vec, extracted_int) = extract_row_according_to_randint_index(feat_matrix, label_vec, batch_size)
        grad_f = objective_obj.mini_batch_grad(x, extracted_matrix, extracted_vec)

        # store
        x_arr = np.vstack((x_arr, x))        
        objective_arr = np.append(objective_arr, objective_new)
        eta_arr = np.append(eta_arr, eta)

        iteration += 1

    return x_arr, objective_arr, eta_arr



if __name__ == "__main__":


    x = np.array([1, -2 , 2])
    C = np.identity(3)
    z = np.dot(x,C)
    print(z)

    pass
