import numpy as np


def accelerated_proximal_gradient_algorithm(objective_obj, prox_operator_obj, line_search_obj, x_init, iter_end = 6000):

    """
    加速付き近接勾配法(Prox-AG)のアルゴリズム

    Parameters
    ----------    
    objective_obj : class[ObjectiveFunction の具象クラス]
        目的関数の抽象クラスの具象クラスのインスタンス
    prox_operator_obj : class[ProximalOperator の具象クラス]
        近接作用素の抽象クラスの具象クラスのインスタンス
        具象クラス名の末尾が ForProximal であるものを想定
    line_search_obj : class[LineSearch の具象クラス]
        ラインサーチの抽象クラスの具象クラスのインスタンス
    x_init : ndarray[float64]
        初期値
        要素数は objective_obj.objective_func(x) の x と同じ
    iter_end : int
        反復回数

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
    eta = line_search_obj.line_search(x_init)

    # store
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    objective_arr = np.append(empty_arr, objective_init)
    eta_arr = np.append(empty_arr, eta)
    

    # iteration = 1,2,....
    iteration = 1
    x = x_init
    y = x_init   # y1 = x0
    t = 1        # t1 = 1
    x_old = x_init
    while iteration <= iter_end:

        # current iteration
        x = prox_operator_obj.proximal_mapping( y - eta * objective_obj.full_path_grad(y), eta)
        t_new = (1 + np.sqrt( 1 + 4 * t**2)) / 2
        y_new = x + (t - 1) / t_new * (x - x_old)
       
       # for next iteration
        x_old = x
        t = t_new
        y = y_new
        eta = line_search_obj.line_search(y)

        # store
        x_arr = np.vstack((x_arr, x))
        objective_new = objective_obj.objective_func(x)
        objective_arr = np.append(objective_arr, objective_new)
        eta_arr = np.append(eta_arr, eta)

        iteration += 1

    return x_arr, objective_arr, eta_arr



if __name__ == "__main__":
    pass
