import numpy as np


def admm_for_lasso_algorithm(objective_obj, prox_operator_obj, rho, x_init, z_init, u_init, iter_end = 6000):

    """
    交互方向乗数法(ADMM)のアルゴリズム
    最小二乗問題 + L1正則化 用

    Parameters
    ----------    
    objective_obj : class[ObjectiveFunction の具象クラス]
        目的関数の抽象クラスの具象クラスのインスタンス
    prox_operator_obj : class[ProximalOperator の具象クラス]
        近接作用素の抽象クラスの具象クラスのインスタンス
        具象クラス名の末尾が ForProximal であるものを想定
    rho : ndarray[float64]
        罰則パラメータ(penalty parameter)
    x_init : ndarray[float64]
        初期値
        要素数は objective_obj.objective_func(x) の x と同じ
    z_init : ndarray[float64]
        初期値
    u_init : ndarray[float64]
        初期値
    iter_end : int
        反復回数

    Returns
    -------
    x_arr : ndarray[float64, float64]
        点 x の計算結果
        axis 0: 反復回数
        axis 1: データ属性
    z_arr : ndarray[float64, float64]
        点 z の計算結果
    u_arr : ndarray[float64, float64]
        点 u の計算結果
    objective_arr : ndarray[float64]
        目的関数 objective(x) の計算結果
        axis 0: 反復回数
                    
    """
    A = objective_obj.get_A()
    b = objective_obj.get_b()
    mod_A_inv = np.linalg.inv(np.dot(A.T, A) + rho * np.identity(A.shape[1]))

    # iteration = 0
    x = x_init
    z = z_init
    u = u_init

    objective_init = objective_obj.objective_func(x)
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    z_arr = np.append(empty_arr, z_init)
    u_arr = np.append(empty_arr, u_init)
    objective_arr = np.append(empty_arr, objective_init)

    # iteration = 1,2,....
    iteration = 1

    while 1:  
        x_new = np.dot(mod_A_inv, np.dot(A.T, b) + rho * (z - u))
        z_new = prox_operator_obj.proximal_mapping(x_new + u, 1 / rho)
        u_new = u + x_new - z_new

        x_arr = np.vstack((x_arr, x_new))
        z_arr = np.vstack((z_arr, z_new))
        u_arr = np.vstack((u_arr, u_new))
        objective_new = objective_obj.objective_func(x_new)
        objective_arr = np.append(objective_arr, objective_new)      

        iteration += 1

        if iteration == iter_end:
            break
        
        x = x_new
        z = z_new
        u = u_new

    return x_arr, z_arr, u_arr, objective_arr



if __name__ == "__main__":
    pass
