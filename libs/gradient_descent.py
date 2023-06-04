import numpy as np


def gradient_descent_algorithm(objective_obj, line_search_obj, x_init, tol = 1e-1):
    
    """
    勾配降下法(GD)のアルゴリズム
   
    Parameters
    ----------
    objective_obj : class[ObjectiveFunction の具象クラス]
        目的関数の抽象クラスの具象クラスのインスタンス
    line_search_obj : class[LineSearch の具象クラス]
        ラインサーチの抽象クラスの具象クラスのインスタンス
    x_init : ndarray[float64]
        初期値
        要素数は objective_obj の feat_matrix の axis 1 と同じ
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

    # iteration = 0
    objective_init = objective_obj.objective_func(x_init)
    delta_x = objective_obj.full_path_grad(x_init)
    eta = line_search_obj.line_search(x_init, delta_x)

    # store
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    objective_arr = np.append(empty_arr, objective_init)
    eta_arr = np.append(empty_arr, eta)
    
    # iteration = 1,2,...
    x = x_init
    while np.linalg.norm(delta_x, ord=2) >= tol:            

        # current iteration
        x_new = x - eta * delta_x
        objective_new = objective_obj.objective_func(x_new)

        # for next iteration
        x = x_new
        delta_x = objective_obj.full_path_grad(x)
        eta = line_search_obj.line_search(x, delta_x)         

        # store        
        x_arr = np.vstack((x_arr, x))
        objective_arr = np.append(objective_arr, objective_new)
        eta_arr = np.append(eta_arr, eta) 

    return x_arr, objective_arr, eta_arr



if __name__ == "__main__":
    pass
