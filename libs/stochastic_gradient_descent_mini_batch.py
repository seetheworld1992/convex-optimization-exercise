import numpy as np
from extract_data_uniform import extract_row_according_to_randint_index


def mini_batch_sgd_algorithm(objective_obj, step_size_rule_obj, x_init, batch_size = 20, iter_end = 6000):
    """
    ミニバッチ確率的勾配法(Mini-Batch SGD)のアルゴリズム
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
        引数の要素数は data_matrix の axis 1 と同じ
    batch_size : int
        勾配計算に用いるバッチ数
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
    # iteration = 0
    objective_init = objective_obj.objective_func(x_init)
    (extracted_matrix, extracted_vec, extracted_int) = extract_row_according_to_randint_index(objective_obj.get_feat_matrix(), objective_obj.get_label_vec(), batch_size)
    delta_x = objective_obj.mini_batch_grad(x_init, extracted_matrix, extracted_vec)
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
        (extracted_matrix, extracted_vec, extracted_int) = extract_row_according_to_randint_index(objective_obj.get_feat_matrix(), objective_obj.get_label_vec(), batch_size)
        delta_x = objective_obj.mini_batch_grad(x, extracted_matrix, extracted_vec)
        eta = step_size_rule_obj.step_size_rule(delta_x, iteration)

        # store
        x_arr = np.vstack((x_arr, x))
        objective_arr = np.append(objective_arr, objective_new)
        eta_arr = np.append(eta_arr, eta) 

        iteration += 1

    return x_arr, objective_arr, eta_arr




if __name__ == "__main__":
    pass
