import numpy as np


def extract_row_according_to_randint_index(data_matrix, label_vec, batch_size = 1):
    """
    一様分布に従うインデックスを生成しそのインデックスの行を抽出する関数
    抽出する行数は batch_size で指定
    batch_size = 1 のとき extracted_mat は(二次元ではなく)一次元配列
    batch_size = 1 のとき extracted_vec, extracted_int は一次元配列のまま(要素数は1)

    Parameters
    ----------
    data_matrix : ndarray[float64, float64]
        axis 0: データサンプル
        axis 1: データ属性
    label_vec : ndarray[float64]
        データサンプルに対応するラベルの配列
        要素数は data_matrix の axis 0 と同じ
    batch_size : int
        抽出するインデックス数

    Returns
    -------
    extracted_mat : ndarray[float64, float64] or ndarray[float64]
        抽出するデータ行列
        axis 0: 抽出するインデックス
        axis 1: データ属性
    extracted_vec : ndarray[float64]
        抽出するラベル
        axis 0: 抽出するインデックス
    extracted_int : ndarray[int]
        抽出するインデックスの配列
        axis 0: 抽出するインデックス
                    
    """

    z = np.random.randint(0, data_matrix.shape[0])
    extracted_mat = data_matrix[z, :]
    extracted_vec = np.empty(0)
    extracted_int = np.empty(0)
    extracted_vec = np.append(extracted_vec, label_vec[z])
    extracted_int = np.append(extracted_int, z)

    for i in range(1, batch_size):
        z = np.random.randint(0, data_matrix.shape[0])
        extracted_mat = np.vstack((extracted_mat, data_matrix[z, :]))
        extracted_vec = np.append(extracted_vec, label_vec[z])
        extracted_int = np.append(extracted_int, z)

    return extracted_mat, extracted_vec, extracted_int


if __name__ == "__main__":
    pass
