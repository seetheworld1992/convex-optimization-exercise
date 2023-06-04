import numpy as np
from abc import ABCMeta, abstractmethod


class ProximalOperator(metaclass=ABCMeta):
    """
    近接写像の抽象クラス
    """

    @abstractmethod
    def proximal_mapping(self, x : np.ndarray):
        pass


class ProximalOperatorL1Norm(ProximalOperator):
    """
    近接写像の具象クラス L1正則化

    Attributes:
        後で書く

    """

    def __init__(self, lamb):
        """
        コンストラクタ

        Args:
            lamb (float64)  : L1正則化項の係数
        """
        self.lamb = lamb
        pass

    def proximal_mapping(self, x, t = 1):
        """
        近接写像を返す関数

        Args:
            x (ndarray[float64]): 入力変数
            t (float64)         : prox_{t h(x)} の t に対応する定数
        Returns:
            ndarray[float64] : prox(x) 

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
        
        self.x = x
        self.t = t

        return self.__soft_thresholding_operator()

    def __soft_thresholding_operator(self):
        x_hat = np.zeros(self.x.shape[0])
        t_lamb = self.t * self.lamb
        x_hat[ self.x >= t_lamb] = self.x[ self.x >= t_lamb] - t_lamb
        x_hat[ self.x <= - t_lamb] = self.x[ self.x <= - t_lamb] + t_lamb
        return x_hat


class ProximalOperatorLeastL1Norm(ProximalOperator):
    """
    近接写像の具象クラス Ax - b のL1ノルム

    Attributes:
        後で書く

    """

    def __init__(self, A, b, lamb = 1):
        """
        コンストラクタ        
        Args:
            A : ndarray[float64, float64]
            b : ndarray[float64]
                要素数は A の axis 0 と同じ
            lamb (float64)       : 正則化項の重み係数
        """
        self.A = A
        self.b = b
        self.lamb = lamb


    def proximal_mapping(self, x, t = 1):
        """
        近接写像を返す関数

        Args:
            x (ndarray[float64]): 入力変数
            t (float64)         : prox_{t h(x)} の t に対応する定数
        Returns:
            ndarray[float64] : prox(x) 

        Raises:
            例外の名前: 例外の説明 (例 : 引数が指定されていない場合に発生 )

        Yields:
            戻り値の型: 戻り値についての説明

        Examples:

        Note:
            注意事項などを記載

        """
        
        self.x = x
        self.t = t

        return self.__soft_thresholding_operator()

    def __soft_thresholding_operator(self):
        # x_hat = np.zeros(self.A.shape[0])        
        x_hat = np.dot(self.A, self.x) - self.b
        t_lamb = self.t * self.lamb
        x_hat[ x_hat >= t_lamb] = x_hat[ x_hat >= t_lamb] - t_lamb
        x_hat[ x_hat <= - t_lamb] = x_hat[ x_hat <= - t_lamb] + t_lamb
        return x_hat


class ProximalOperatorZero(ProximalOperator):
    """
    近接写像の具象クラス ゼロ

    Attributes:
        後で書く

    """

    def __init__(self, lamb = 0):
        pass

    def proximal_mapping(self, x, t = 1):
        """
        近接写像を返す関数

        Args:
            x (ndarray[float64]): 入力変数
            t (float64)         : prox_{t h(x)} の t に対応する定数
        Returns:
            ndarray[float64] : 0

        Note:
            引数 t は消さないこと

        """
        return x


if __name__ == "__main__":
    
    x = np.array([2, -1/2 ,-3])
    gamma = 1
    prox = ProximalOperatorL1Norm(gamma)
    proxz = ProximalOperatorZero(gamma)
    print(proxz.proximal_mapping(x))