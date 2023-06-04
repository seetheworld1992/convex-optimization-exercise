import numpy as np
from numpy.core.numeric import identity
from numpy.lib.function_base import iterable
from extract_data_uniform import extract_row_according_to_randint_index

def update_l_bfgs_online(grad_f, s_arr, y_arr):

    q = np.copy(grad_f)
    memory_size = s_arr.shape[0]
    sum = 0
    for i in range(memory_size - 1, -1, -1): # i = m-1, m-2, .., 1, 0
        s_i = s_arr[i,:]
        y_i = y_arr[i,:]
        sum += np.dot(s_i, y_i) / np.dot(y_i, y_i)
    
    q *= sum / memory_size

    return q


def stochastic_l_bfgs_method_algorithm(objective_obj, step_size_rule_obj, x_init, batch_size = 20, c = 0.1, lambd = 1, epsilon = 1e-10, m = 10, iter_end = 6000):
    
    """
    確率的準ニュートン法 LBFGS のアルゴリズム
    Nicol N. Schraudolph, Jin Yu, Simon G¨unter の
    A Stochastic Quasi-Newton Method for Online Convex Optimization のアルゴリズム oLBFGS を実装
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
    m : int
        メモリに保持するイタレーション数 (m > 2 とすること)
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
    dim = feat_matrix.shape[1]

    # iteration = 0    
    objective_init = objective_obj.objective_func(x_init)
    (extracted_matrix, extracted_vec, extracted_int) = extract_row_according_to_randint_index(objective_obj.get_feat_matrix(), objective_obj.get_label_vec(), batch_size)
    grad_f = objective_obj.mini_batch_grad(x_init, extracted_matrix, extracted_vec)
    minus_l_bfgs_step = epsilon * grad_f
    iteration = 0
    eta = step_size_rule_obj.step_size_rule(minus_l_bfgs_step, iteration)
    
    # store    
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    objective_arr = np.append(empty_arr, objective_init)
    eta_arr = np.append(empty_arr, eta)
    s_arr = np.append(empty_arr, np.array(np.zeros(dim)) )
    y_arr = np.append(empty_arr, np.array(np.zeros(dim)) )

    # iteration = 1
    x = x_init
    x_new = x - eta * minus_l_bfgs_step   # x_1
    objective_new = objective_obj.objective_func(x_new)
    grad_f_new = objective_obj.mini_batch_grad(x_new, extracted_matrix, extracted_vec)     # grad_f_1

    # for memory limited
    s = - eta * minus_l_bfgs_step   # = x_new - x = s_0
    y = grad_f_new - grad_f         # y_0    
    s_arr = np.vstack((s_arr, s))
    y_arr = np.vstack((y_arr, y))
    s_arr = np.delete(s_arr, 0, 0)
    y_arr = np.delete(y_arr, 0, 0)

    # for next iteration
    x = x_new
    minus_l_bfgs_step = update_l_bfgs_online(grad_f_new, s_arr, y_arr)   # ← s_0, y_0
    eta = step_size_rule_obj.step_size_rule(minus_l_bfgs_step, iteration)
    (extracted_matrix, extracted_vec, extracted_int) = extract_row_according_to_randint_index(feat_matrix, label_vec, batch_size)
    grad_f = objective_obj.mini_batch_grad(x, extracted_matrix, extracted_vec)

    # store
    x_arr = np.vstack((x_arr, x))    
    objective_arr = np.append(objective_arr, objective_new)
    eta_arr = np.append(eta_arr, eta)

    # iteration = 2,3,...
    iteration += 1
    x = x_init
    while iteration <= iter_end:

        # current iteration
        x_new = x - eta * minus_l_bfgs_step
        objective_new = objective_obj.objective_func(x_new)
        grad_f_new = objective_obj.mini_batch_grad(x_new, extracted_matrix, extracted_vec)

        # for memory limited        
        # s = - eta * minus_l_bfgs_step  #  = x_new - x
        # y = grad_f_new - grad_f
        s = - eta * minus_l_bfgs_step / c  # =  ( x_new - x ) / c
        y = grad_f_new - grad_f + lambd * s
        s_arr = np.vstack((s_arr, s))
        y_arr = np.vstack((y_arr, y))
        if iteration > m:
            s_arr = np.delete(s_arr, 0, 0)
            y_arr = np.delete(y_arr, 0, 0)    

        # for next iteration
        x = x_new
        minus_l_bfgs_step = update_l_bfgs_online(grad_f_new, s_arr, y_arr)
        eta = step_size_rule_obj.step_size_rule(minus_l_bfgs_step, iteration)
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
