import numpy as np
from numpy.core.numeric import identity
from numpy.lib.function_base import iterable


def update_bfgs(H, s, y):
    rho = 1 / np.dot(y,s)
    vec_Hy = np.dot(H, y)
    H_new = H - rho * np.outer(vec_Hy, s) - rho * np.outer( s, np.dot(y, H) ) + rho * rho * np.outer( s , np.dot(y, vec_Hy) * s) + rho * np.outer(s, s)
    
    # I = identity(H.shape[0])
    # tmp_s_y = I - rho * np.outer(s, y) 
    # tmp_y_s = I - rho * np.outer(y, s)
    # H_new = np.dot( tmp_s_y , H - rho * np.outer(Hy,s) ) + rho * np.outer(s, s) 
    # H_new = np.dot(tmp_s_y, np.dot(H, tmp_y_s) ) + rho * np.outer(s, s)

    return H_new


def bfgs_method_algorithm(objective_obj, line_search_obj, x_init, tol = 1e-1):
    
    """
    準ニュートン法 BFGS のアルゴリズム
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
    grad_f = objective_obj.full_path_grad(x_init)
    H_init = np.identity(x_init.shape[0])
    minus_bfgs_step = np.dot(H_init, grad_f)
    eta = line_search_obj.line_search(x_init, minus_bfgs_step)
    
    # store    
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    objective_arr = np.append(empty_arr, objective_init)
    eta_arr = np.append(empty_arr, eta)

    # iteration = 1,2,...
    x = x_init
    first_time = True
    while np.linalg.norm(grad_f, ord=2) >= tol:

        # current iteration
        x_new = x - eta * minus_bfgs_step
        objective_new = objective_obj.objective_func(x_new)
        grad_f_new = objective_obj.full_path_grad(x_new)        

        # for next iteration       
        s = - eta * minus_bfgs_step  # = x_new - x
        y = grad_f_new - grad_f        
        if first_time:
            Hessian_inv = np.dot(y,s) / np.dot(y,y) * H_init
            first_time = False
        Hessian_inv = update_bfgs(Hessian_inv, s, y)
        x = x_new
        grad_f = grad_f_new
        minus_bfgs_step = np.dot(Hessian_inv, grad_f)
        eta = line_search_obj.line_search(x, minus_bfgs_step)

        # store
        x_arr = np.vstack((x_arr, x))        
        objective_arr = np.append(objective_arr, objective_new)
        eta_arr = np.append(eta_arr, eta)

    return x_arr, objective_arr, eta_arr



if __name__ == "__main__":


    x = np.array([1, -2 , 2])
    C = np.identity(3)
    z = np.dot(x,C)
    print(z)

    pass
