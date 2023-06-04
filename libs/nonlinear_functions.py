import numpy as np


class SigmoidFunction:
    """
    シグモイド関数クラス

    Attributes:
        後で書く

    """

    def __init__(self):
        pass

    def logistic_function(self, x):
        """
        ロジスティック関数を返す関数

        Args:
            x (ndarray[float64]): 入力変数
        Returns:
            ndarray[float64] : logistic_function(x)
        """
        
        self.x = x
        
        return 1 / ( 1 + np.exp( - self.x))


if __name__ == "__main__":
    
    x = np.array([2, -1/2 ,-3])
    sig = SigmoidFunction()
    print(sig.logistic_function(x))