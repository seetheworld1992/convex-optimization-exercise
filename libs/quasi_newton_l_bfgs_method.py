import numpy as np


def update_l_bfgs(grad_f, s_arr, y_arr):

    q = np.copy(grad_f)
    memory_size = s_arr.shape[0]
    alpha = np.zeros(memory_size)
    for j in range(memory_size - 1, -1, -1): # j = m-1, m-2, .., 1, 0
        s_i = s_arr[j,:]
        y_i = y_arr[j,:]
        alpha[j] = np.dot(s_i, q) / np.dot(y_i, s_i)
        q -= alpha[j] * y_i

    s = s_arr[memory_size - 1, :]
    y = y_arr[memory_size - 1, :]
    q *= np.dot(s,y) / np.dot(y,y)
    for j in range(memory_size): # j = 0, 1, .., m-2, m-1
        s_i = s_arr[j,:]
        y_i = y_arr[j,:]
        beta = np.dot(y_i, q) / np.dot(y_i, s_i)
        q += (alpha[j] - beta) * s_i

    return q


def l_bfgs_method_algorithm(objective_obj, line_search_obj, x_init, H_init, m = 10, tol = 1e-1):
    
    """
    準ニュートン法 L-BFGS のアルゴリズム
    Nocedal and Wright, Numerical Optimization のアルゴリズムを実装
   
    Parameters
    ----------
    objective_obj : class[ObjectiveFunction の具象クラス]
        目的関数の抽象クラスの具象クラスのインスタンス
    line_search_obj : class[LineSearch の具象クラス]
        ラインサーチの抽象クラスの具象クラスのインスタンス
    x_init : ndarray[float64]
        初期値
        要素数は objective_obj の feat_matrix の axis 1 と同じ
    H_init : ndarray[float64, float64]
        初期値
        行と列の要素数は objective_obj の feat_matrix の axis 1 と同じ
    m : int
        メモリに保持するイタレーション数 (m > 2 とすること)
    tol : float64
        絶対許容誤差

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

    dim = x_init.shape[0]

    # iteration = 0
    objective_init = objective_obj.objective_func(x_init)
    grad_f = objective_obj.full_path_grad(x_init)                  # grad_f_0
    minus_l_bfgs_step = np.dot(H_init, grad_f)                     # - p_0
    eta = line_search_obj.line_search(x_init, minus_l_bfgs_step)   # eta_0    

    # store
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    objective_arr = np.append(empty_arr, objective_init)
    eta_arr = np.append(empty_arr, eta)
    s_arr = np.append(empty_arr, np.array(np.zeros(dim)) )
    y_arr = np.append(empty_arr, np.array(np.zeros(dim)) )

    # iteration = 1
    x = x_init
    x_new = x - eta * minus_l_bfgs_step                 # x_1
    objective_new = objective_obj.objective_func(x_new)
    grad_f_new = objective_obj.full_path_grad(x_new)    # grad_f_1

    # for memory limited
    s = - eta * minus_l_bfgs_step                       # = x_new - x = s_0
    y = grad_f_new - grad_f                             # y_0    
    s_arr = np.vstack((s_arr, s))
    y_arr = np.vstack((y_arr, y))
    s_arr = np.delete(s_arr, 0, 0)
    y_arr = np.delete(y_arr, 0, 0)
    
    # for next iteration
    x = x_new
    grad_f = grad_f_new
    minus_l_bfgs_step = update_l_bfgs(grad_f, s_arr, y_arr)   # ← s_0, y_0
    eta = line_search_obj.line_search(x, minus_l_bfgs_step)

    # store
    x_arr = np.vstack((x_arr, x))    
    objective_arr = np.append(objective_arr, objective_new)
    eta_arr = np.append(eta_arr, eta)

    # iteration = 2,3,...
    iteration = 2
    while np.linalg.norm(grad_f, ord=2) >= tol:

        # current iteration
        x_new = x - eta * minus_l_bfgs_step
        objective_new = objective_obj.objective_func(x_new)
        grad_f_new = objective_obj.full_path_grad(x_new)

        # for memory limited
        s = - eta * minus_l_bfgs_step  #  = x_new - x
        y = grad_f_new - grad_f
        s_arr = np.vstack((s_arr, s))
        y_arr = np.vstack((y_arr, y))
        if iteration > m:
            s_arr = np.delete(s_arr, 0, 0)
            y_arr = np.delete(y_arr, 0, 0)

        # for next iteration
        x = x_new
        grad_f = grad_f_new
        minus_l_bfgs_step = update_l_bfgs(grad_f, s_arr, y_arr)        
        eta = line_search_obj.line_search(x, minus_l_bfgs_step)

        # store
        x_arr = np.vstack((x_arr, x))        
        objective_arr = np.append(objective_arr, objective_new)
        eta_arr = np.append(eta_arr, eta)

        iteration += 1

    return x_arr, objective_arr, eta_arr



if __name__ == "__main__":


    num = 4
    print(np.arange(num)[::-1])

    memory_size = 3
    for j in range(memory_size - 1, -1, -1):
        print(j)
    
    for j in range(memory_size):
        print(j)

    pass
