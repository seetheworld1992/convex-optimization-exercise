import numpy as np
import matplotlib.pyplot as plt
import sys, os
sys.path.append(os.path.join('..', 'libs'))

from objective_functions import LogisticLoss
from proximal_operators import ProximalOperatorL1Norm
from proximal_gradient import proximal_gradient_algorithm
from line_searches import ConstantLineSearch

#
# 目的関数      : ロジスティックロス関数
# 最適化        : 近接勾配法
# ステップサイズ : 固定ステップ (データ行列の特異値分解を利用)
#
# 甲斐性なしのブログ 
# 近接勾配法応用編その2 ～L1ノルム正則化項によるスパースなロジスティック回帰～ の追試
# https://yamagensakam.hatenablog.com/entry/2018/06/03/100542
# 
# L1正則化項の係数 (lamb * サンプルサイズ)  の Lamb を 0,1,2,...,40 と変えてその影響をみる
#

print("Hello")

# データを作成する
np.random.seed(seed=0)    
sample_size = 400   # サンプル数
sample_size_half =  int(sample_size/2)
feature_num = 125  # 特徴量の数

# 訓練データ    
# 最初の2次元(0 列目と1 列目)だけ分類に寄与する属性
X_train_1 = np.random.randn(sample_size_half,2)
X_train_2 = np.ones((sample_size_half,2)) + np.random.randn(sample_size_half,2)
X_train = np.concatenate([X_train_1, X_train_2], 0)
X_train_3 = np.random.randn(sample_size, feature_num - 2)
X_train = np.concatenate([X_train, X_train_3], 1)

# 訓練データ ラベル
y_train1 = np.ones((sample_size_half, 1))
y_train2 = np.zeros((sample_size_half, 1))
y_train = np.concatenate([y_train1, y_train2], 0).squeeze()

# テストデータを作成する    
# テストデータ
X_test_1 = np.zeros((sample_size_half,2)) + np.random.randn(sample_size_half,2)
X_test_2 = np.ones((sample_size_half,2)) + np.random.randn(sample_size_half,2)
X_test = np.concatenate([X_test_1,X_test_2], 0)
X_test_3 = np.random.randn(sample_size, feature_num - 2)

X_test = np.concatenate([X_test, X_test_3], 1)

# テストデータ ラベル
y_test = y_train

data_matrix = X_train
label_vec = y_train

data_matrix = np.concatenate([np.ones((data_matrix.shape[0], 1)), data_matrix], 1)  # 先頭の列にバイアス項を追加 (列の次元が +1 される)


(u, l, v) = np.linalg.svd(data_matrix.T)
eta = 1.0 / max(l.real*l.real)
constant_ls_obj = ConstantLineSearch(eta)
iteration_end = 100
x_init = np.zeros(data_matrix.shape[1])

cnt_max_plus_one = 41
non_zero_cnt_arr = np.zeros(cnt_max_plus_one)
acc_arr = np.zeros(cnt_max_plus_one)
for i in range(0, cnt_max_plus_one):
    lamb = i
    logistic_obj = LogisticLoss(data_matrix, label_vec, 'l1', lamb * sample_size)
    prox_1norm_obj = ProximalOperatorL1Norm(lamb)
    (x_arr, loss_arr, eta_arr) = proximal_gradient_algorithm(logistic_obj, prox_1norm_obj, constant_ls_obj, x_init, iteration_end)

    x_arr_last = x_arr[x_arr.shape[0] - 1,:]
    y_result = np.dot(x_arr_last[1:feature_num + 1], X_test.T) + x_arr_last[0]
    y_result[y_result >= 0] = 1
    y_result[y_result < 0] = 0
    non_zero_cnt_arr[lamb] = np.sum(np.abs(x_arr_last) > 1e-12)
    acc_arr[lamb] = np.sum(y_result == y_test)/float(y_test.shape[0])

fig, ax1 = plt.subplots()
ax1.plot(acc_arr, "r.-")
ax1.set_ylabel("accuracy")
ax2=plt.twinx()
ax2.plot(non_zero_cnt_arr, "bx-")
ax2.set_ylabel("non zero element count")
plt.legend()
plt.show()
