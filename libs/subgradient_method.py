import numpy as np


def subgradient_algorithm(objective_obj, step_size_rule_obj, x_init, iter_end = 6000):
    
    """
    劣勾配法(SG)のアルゴリズム
   
    Parameters
    ----------
    objective_obj : class[ObjectiveFunction の具象クラス]
        目的関数の抽象クラスの具象クラスのインスタンス
    step_size_rule_obj : class[LineSearch の具象クラス]
        ステップサイズルールの抽象クラスの具象クラスのインスタンス
    x_init : ndarray[float64]
        初期値
        引数の要素数は objective_obj の feat_matrix の axis 1 と同じ
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
    delta_x = objective_obj.full_path_subgrad(x_init)    # subgradient
    iteration = 0
    eta = step_size_rule_obj.step_size_rule(delta_x, iteration)

    # store
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    objective_arr = np.append(empty_arr, objective_init)    
    eta_arr = np.append(empty_arr, eta)

    # iteration = 1,2,....
    iteration += 1
    x = x_init
    while iteration <= iter_end:

        # current iteration
        x_new = x - eta * delta_x
        objective_new = objective_obj.objective_func(x_new)

        # for next iteration
        x = x_new
        delta_x = objective_obj.full_path_subgrad(x)    # subgradient
        eta = step_size_rule_obj.step_size_rule(delta_x, iteration)

        # store
        x_arr = np.vstack((x_arr, x))        
        objective_arr = np.append(objective_arr, objective_new)
        eta_arr = np.append(eta_arr, eta)

        iteration += 1

    return x_arr, objective_arr, eta_arr



if __name__ == "__main__":
    pass
