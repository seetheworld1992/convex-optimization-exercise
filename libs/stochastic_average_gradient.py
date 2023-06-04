import numpy as np
from extract_data_uniform import extract_row_according_to_randint_index


def stochastic_average_gradient_algorithm(objective_obj, step_size_rule_obj, x_init, iter_end = 6000):
    """
    確率的平均勾配法(SAG)のアルゴリズム
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
    sample_size = feat_matrix.shape[0]
    batch_size = 1

    # iteration = 0
    objective_init = objective_obj.objective_func(x_init)
    delta_x_init = objective_obj.full_path_grad(x_init)
    iteration = 0
    eta = step_size_rule_obj.step_size_rule(delta_x_init, iteration)
    
    grad_storage_matrix = objective_obj.one_path_grad(x_init, feat_matrix[0, :], label_vec[0])
    for i in range(1, sample_size):
        z = objective_obj.one_path_grad(x_init, feat_matrix[i, :], label_vec[i])
        grad_storage_matrix = np.vstack([grad_storage_matrix, z])    # grad_storage_mat  行: データサンプル数  列: データ属性
        
    # store
    empty_arr = np.empty(0)
    x_arr = np.append(empty_arr, x_init)
    objective_arr = np.append(empty_arr, objective_init)
    eta_arr = np.append(empty_arr, eta)

    # iteration = 1,2,....
    iteration += 1
    x = x_init
    table_ave = delta_x_init 
    while iteration <= iter_end:

        # current iteration
        x_new = x - eta * table_ave
        objective_new = objective_obj.objective_func(x_new)

        # for next iteration
        x = x_new
        table_ave_old = table_ave        
        (extracted_matrix, extracted_vec, extracted_int) = extract_row_according_to_randint_index(feat_matrix, label_vec, batch_size)
        grad_draw = objective_obj.one_path_grad(x, extracted_matrix, extracted_vec)
        table_ave = ( grad_draw - grad_storage_matrix[int(extracted_int), :] ) / sample_size + table_ave_old        
        grad_storage_matrix[int(extracted_int), :] = grad_draw
        eta = step_size_rule_obj.step_size_rule(x, iteration)

        # store
        x_arr = np.vstack((x_arr, x_new))
        objective_arr = np.append(objective_arr, objective_new)
        eta_arr = np.append(eta_arr, eta)       

        iteration += 1

    return x_arr, objective_arr, eta_arr



if __name__ == "__main__":
    pass
