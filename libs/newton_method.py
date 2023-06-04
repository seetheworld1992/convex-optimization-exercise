import numpy as np


def newton_method_algorithm(objective_obj, line_search_obj, x_init, tol = 1e-1):
    
    """
    ニュートン法のアルゴリズム
    Boyd and Vandenberghe, Convex Optimization のアルゴリズムを実装
   
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
        目的関数 objective(x) の計算結果
        axis 0: 反復回数
    eta_arr : ndarray[float64]
        ステップサイズ eta の計算結果
        axis 0: 反復回数
                    
    """

    # iteration = 0
    objective_init = objective_obj.objective_func(x_init)
    grad_f = objective_obj.full_path_grad(x_init)
    minus_newton_step = np.linalg.solve(objective_obj.hessian_matrix(x_init), grad_f)
    lambda_square = np.dot(grad_f, minus_newton_step)
    eta = line_search_obj.line_search(x_init, minus_newton_step)

    # store
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    objective_arr = np.append(empty_arr, objective_init)
    eta_arr = np.append(empty_arr, eta)

    # iteration = 1,2,...
    x = x_init
    while lambda_square > 2 * tol:

        # current iteration
        x_new = x - eta * minus_newton_step
        objective_new = objective_obj.objective_func(x_new)

        # for next iteration
        x = x_new
        grad_f = objective_obj.full_path_grad(x)
        minus_newton_step = np.linalg.solve(objective_obj.hessian_matrix(x), grad_f)
        lambda_square = np.dot(grad_f, minus_newton_step)
        eta = line_search_obj.line_search(x, minus_newton_step)

        # store
        x_arr = np.vstack((x_arr, x_new))        
        objective_arr = np.append(objective_arr, objective_new)
        eta_arr = np.append(eta_arr, eta)        

    return x_arr, objective_arr, eta_arr



if __name__ == "__main__":
    pass
