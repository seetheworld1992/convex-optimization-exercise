import numpy as np
from abc import ABCMeta, abstractmethod
from nonlinear_functions import SigmoidFunction


class ObjectiveFunction(metaclass=ABCMeta):
    """
    目的関数の抽象クラス
    """

    @abstractmethod
    def objective_func(self, x : np.ndarray):
        pass

    @abstractmethod
    def full_path_grad(self, x : np.ndarray, feat_matrix : np.ndarray, label_vec : np.ndarray):
        pass

    @abstractmethod
    def one_path_grad(self, x : np.ndarray, feat_vec : np.ndarray, label_int : int):
        pass

    @abstractmethod
    def mini_batch_grad(self, x : np.ndarray, feat_matrix : np.ndarray, label_vec : np.ndarray):
        pass

    @abstractmethod
    def hessian_matrix(self, x : np.ndarray, feat_matrix : np.ndarray, label_vec : np.ndarray):
        pass

    @abstractmethod
    def full_path_subgrad(self, x : np.ndarray, feat_matrix : np.ndarray, label_vec : np.ndarray):
        pass


class LeastSquares(ObjectiveFunction):
    """
    具象クラス : 最小二乗問題

    Attributes:
        後で書く

    """

    def __init__(self, A, b, regularization = 'none', lamb = 0):
        """
        コンストラクタ
        Args:
            A : ndarray[float64, float64]
            b : ndarray[float64]
                要素数は A の axis 0 と同じ
            regularization (str) : 正則化の種類  none / l1 を想定
            lamb (float64)       : 正則化項の重み係数

        """

        self.A = A
        self.b = b
        self.regularization = regularization
        self.lamb = lamb

    def objective_func(self, x):

        tmp = np.dot(self.A, x) - self.b
        res = np.dot(tmp, tmp) / 2

        if self.regularization == 'l1':
            res += self.lamb * np.linalg.norm(x, ord=1)

        return res

    def full_path_grad(self, x):
        return  np.dot( self.A.T, np.dot(self.A, x) - self.b)  
        
    def one_path_grad(self):
        return 0

    def hessian_matrix(self, x):
        return np.dot(self.A.T , self.A)

    def full_path_subgrad(self, x):
        return self.full_path_grad(self, x)
    
    def mini_batch_grad(self, x):
        # 使う時に作る
        return 0

    def get_A(self):
        return self.A

    def get_b(self):
        return self.b


class LogisticLoss(ObjectiveFunction):
    """
    具象クラス : ロジスティックロス関数 

    Attributes:
        後で書く

    """

    def __init__(self, feat_matrix, label_vec, regularization = 'none', lamb = 0):
        """
        コンストラクタ
        Args:
            feat_matrix : ndarray[float64, float64]
                axis 0: データサンプル
                axis 1: データ属性
            label_vec : ndarray[float64]
                データサンプルに対応するラベルの配列
                要素数は feat_matrix の axis 0 と同じ
            regularization (str) : 正則化の種類  none / l1 / l2 を想定
            lamb (float64)       : 正則化項の重み係数

        """
        sig = SigmoidFunction()

        self.feat_matrix = feat_matrix
        self.label_vec = label_vec
        self.regularization = regularization
        self.lamb = lamb
        self.sig = sig
        self.sample_size = self.feat_matrix.shape[0] 
    

    def objective_func(self, w):
        """
        目的関数値を返す関数

        Args:
            w (ndarray[float64]): 入力変数

        Returns:
            res (ndarray[float64]) : objective(w, feat_matrix, label_vec)

        Raises:
            例外の名前: 例外の説明 (例 : 引数が指定されていない場合に発生 )

        Yields:
            戻り値の型: 戻り値についての説明

        Examples:

            関数の使い方について記載

            >>> x = np.array([2, -1/2 ,-3])
                gamma = 1
                prox = ProximalNorm1()
                print(prox.proximal_mapping(x, gamma))

        Note:
            注意事項などを記載

        """
        
        tmp = self.__p_func(w, self.feat_matrix)
        res = - np.sum( self.label_vec * np.log(tmp)  + (1 - self.label_vec) * np.log(1 - tmp) )  / self.sample_size

        # res = 0
        # for i in range(self.sample_size):
        #     res -= self.label_vec[i] * np.dot(self.feat_matrix[i],w)
        #     res += np.log(1 + np.exp(np.dot(self.feat_matrix[i],w)) )

        # res /= self.sample_size

        
        if self.regularization == 'l1': 
            res += self.lamb * np.linalg.norm(w, ord=1)
        elif self.regularization == 'l2':
            res += 0.5 * self.lamb * np.dot(w, w) 

        return res

    def full_path_grad(self, w):
        """
        目的関数の勾配を返す関数
        Args:
            w (ndarray[float64]): 入力変数

        Returns:
            res (ndarray[float64]) : full_path_grad(w)  要素数は feat_matrix の axis 1 と同じ
         """

        res = np.dot(self.feat_matrix.T , self.__p_func(w, self.feat_matrix) - self.label_vec)  / self.sample_size

        if self.regularization == 'l2':
            res += self.lamb * w

        return  res


    def one_path_grad(self, w, feat_vec, label_int):
        """
        目的関数のsummationの内部の項の勾配を返す関数
        Args:
            w (ndarray[float64])        : 入力変数
            feat_vec (ndarray[float64]) : self.feat_matrix の全ての行から axis 0 についてある1つの行を選び取り出したもの (ndarray[float64])
            label_int (int)             : self.label_vec の全ての要素からある1つを選び取り出したもの (int)

        Returns:
            res (ndarray[float64]) : one_path_grad(w, feat_vec, label_int)  要素数は feat_matrix の axis 1 と同じ
         """

        res = (self.__p_func(w, feat_vec) - label_int) * feat_vec

        if self.regularization == 'l2':
            res += self.lamb * w

        return res


    def mini_batch_grad(self, w, feat_matrix, label_vec):
        """
        目的関数の勾配を返す関数
        Args:
            w (ndarray[float64]): 入力変数
            feat_matrix (ndarray[float64, float64]) : self.feat_matrix の全ての行から axis 0 についていくつかの行を選び取り出したもの
            label_vec (ndarray[float64])            : self.label_vec の全ての要素からいくつかを選び取り出したもの

        Returns:
            res (ndarray[float64]) : mini_batch_grad(w, feat_matrix, label_vec)  要素数は feat_matrix の axis 1 と同じ
         """

        batch_size = feat_matrix.shape[0]
        res = np.dot(feat_matrix.T , self.__p_func(w, feat_matrix) - label_vec)  / batch_size

        if self.regularization == 'l2':
            res += self.lamb * w

        return res   


    def __p_func(self, w, X):    # X はベクトルでもよい そのとき __p_func はスカラー  

        return self.sig.logistic_function(np.dot(X, w))

    def hessian_matrix(self, x):
        return 0

    def full_path_subgrad(self, x):
        return self.full_path_grad(self, x)

    def get_feat_matrix(self):
        return self.feat_matrix
    
    def get_label_vec(self):
        return self.label_vec
    

class QuadraticFunction(ObjectiveFunction):

    def __init__(self, Q, regularization = 'none', lamb = 0):
        self.Q = Q
        self.regularization = regularization
        self.lamb = lamb

    def objective_func(self, x):
        res = np.dot(x, np.dot(self.Q, x)) / 2        
        if self.regularization == 'l1':
            res += self.lamb * np.linalg.norm(x, ord=1)
        return res

    def full_path_grad(self, x):
        return np.dot(self.Q , x)

    def one_path_grad(self, x):
        return 0
    
    def mini_batch_grad(self, x):
        # 使う時に作る
        return 0
    
    def hessian_matrix(self, x):
        return self.Q

    def full_path_subgrad(self, x):
        return self.full_path_grad(self, x)


class LogSumExpFunction(ObjectiveFunction):

    def __init__(self, A, b, regularization = 'none', lamb = 0):
        self.A = A
        self.b = b
        self.n = self.A.shape[0]
        self.p = self.A.shape[1]
        self.regularization = regularization
        self.lamb = lamb

    def objective_func(self, x):
        res = np.log(np.sum(np.exp(np.dot(self.A,x) + self.b)))
        if self.regularization == 'l1':
            res += self.lamb * np.linalg.norm(x, ord=1)
        return res

    def full_path_grad(self, x):

        numerator = np.zeros(self.p)
        denominator = np.sum(np.exp(np.dot(self.A,x) + self.b))

        for j in range(self.p):
            tmp = 0
            for i in range(self.n):
                tmp = tmp + self.A[i,j] * np.exp(np.dot(self.A[i,:],x) + self.b[i])

            numerator[j] = tmp

        return numerator / denominator

    def one_path_grad(self, x):
        # 使う時に作る
        return 0
    
    def mini_batch_grad(self, x):
        # 使う時に作る
        return 0

    def hessian_matrix(self, x):
        # 使う時に作る
        return 0

    def full_path_subgrad(self, x):
        return self.full_path_grad(self, x)


class LeastL1Norm(ObjectiveFunction):

    def __init__(self, A, b, regularization = 'none', lamb = 0):
        """
        コンストラクタ
        Args:
            A : ndarray[float64, float64]
            b : ndarray[float64]
                要素数は A の axis 0 と同じ
            regularization (str) : 正則化の種類  none / l1 を想定
            lamb (float64)       : 正則化項の重み係数

        """

        self.A = A
        self.b = b
        self.regularization = regularization
        self.lamb = lamb

    def objective_func(self, x):
        res = np.linalg.norm(np.dot(self.A, x) - self.b, ord = 1)

        if self.regularization == 'l1':
            res += self.lamb * np.linalg.norm(x, ord=1)
        return res

    def full_path_grad(self, x):
        # 微分不可能
        return 0

    def one_path_grad(self, x):
        # 微分不可能
        return 0

    def mini_batch_grad(self, x):
         # 微分不可能
        return 0

    def hessian_matrix(self, x):
        # 微分不可能
        return 0

    def full_path_subgrad(self, x):
        res = np.dot(self.A.T, np.sign(np.dot(self.A, x) - self.b) )
        return res


class Zero(ObjectiveFunction):

    def __init__(self,  A, b, regularization = 'none', lamb = 0):
        """
        コンストラクタ
        Args:
            A : ndarray[float64, float64]
            b : ndarray[float64]
                要素数は A の axis 0 と同じ
            regularization (str) : 正則化の種類  none / l1 を想定
            lamb (float64)       : 正則化項の重み係数

        """
        self.A = A
        self.b = b
        self.regularization = regularization
        self.lamb = lamb

    def objective_func(self, x):
        res = 0
        if self.regularization == 'l1':
            res += self.lamb * np.linalg.norm(x, ord=1)

        elif self.regularization == 'least_l1':
            res += self.lamb * np.linalg.norm( np.dot(self.A, x) - self.b, ord = 1)

        return res

    def full_path_grad(self, x):
        return 0

    def one_path_grad(self, x):
        return 0
 
    def hessian_matrix(self, x):
        return 0

    def full_path_subgrad(self, x):
        return 0





if __name__ == "__main__":
    

    D = np.array([[1,3],[-1,2]])
    E = np.array([[-1,4],[2,3]])

    w = np.dot(D.T,E)
    print(w)


    a = np.array([1,-2, 0])
    b = np.array([-2, 2, 5])
    print(a * b)

    print(1 - a)

    pass