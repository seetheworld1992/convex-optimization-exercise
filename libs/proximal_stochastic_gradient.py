import numpy as np
from extract_data_uniform import extract_row_according_to_randint_index


def proximal_stochastic_gradient_algorithm(objective_obj, prox_operator_obj, x_init, eta, iter_end = 6000):

    """
    確率的近接勾配法(Prox-SG)のアルゴリズム
    目的関数が経験損失のようなsummationである場合を想定
    ラインサーチ機能は無し
    ステップサイズは eta (定数)

    Parameters
    ----------
    objective_obj : class[ObjectiveFunction の具象クラス]
        目的関数の抽象クラスの具象クラスのインスタンス
    prox_operator : class[ProximalOperatorの具象クラス]
        近接作用素抽象クラスの具象クラスのインスタンス
    x_init : ndarray[float64]
        初期値
        引数の要素数は data_matrix の axis 1 と同じ
    beta : float64
        ステップサイズの逆数
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
                    
    """
    # iteration = 0
    x = x_init
    objective_init = objective_obj.objective_func(x)
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    objective_arr = np.append(empty_arr, objective_init)

    # iteration = 1,2,....
    eta_coefficient = eta
    iteration = 1

    while 1:
        eta =  eta_coefficient
        (extracted_matrix, extracted_vec, extracted_int) = extract_row_according_to_randint_index(objective_obj.feat_matrix, objective_obj.label_vec)
        
        x_new = prox_operator_obj.proximal_mapping( x - eta * objective_obj.one_path_grad(x, extracted_matrix, extracted_vec) )

        x_arr = np.vstack((x_arr, x_new))
        objective_new = objective_obj.objective_func(x_new)
        objective_arr = np.append(objective_arr, objective_new)

        iteration += 1

        if iteration == iter_end:
            break
        
        x = x_new

    return x_arr, objective_arr



if __name__ == "__main__":
    pass
