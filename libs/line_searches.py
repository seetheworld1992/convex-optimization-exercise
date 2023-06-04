import numpy as np
from abc import ABCMeta, abstractmethod


class LineSearch(metaclass=ABCMeta):
    """
    直線探索の抽象クラス

    """
    @abstractmethod
    def line_search(self, x : np.ndarray, y : np.ndarray):
        pass


class ConstantLineSearch:
    """
    具象クラス : 定数ステップサイズ(ラインサーチはしない)

    Attributes:
        後で書く

    """

    def __init__(self, step_size):
        """
        コンストラクタ
        Args:
            step_size : float64                
        """ 
        self.step_size =  step_size

    def line_search(self, x = 0, y = 0, delta_x = 0):
        return self.step_size


class BacktrackingLineSearch(LineSearch):
    """
    具象クラス : バックトラッキング

    Attributes:
        後で書く

    """

    def __init__(self, objective_obj, alpha = 0.1, beta = 0.7):
        """
        コンストラクタ
        Args:
            objective_obj : class[ObjectiveFunction の具象クラス]
                目的関数の抽象クラスの具象クラスのインスタンス
            alpha : バックトラッキングのパラメータ 通常は 0.01 - 0.3 から選択
            beta  : バックトラッキングのパラメータ 通常は 0.1 - 0.8 から選択

        """ 
        self. objective_obj =  objective_obj
        self.alpha = alpha
        self.beta = beta


    def line_search(self, x, delta_x):        
        t = 1
        fx = self.objective_obj.objective_func(x)
        delta_fx = self.objective_obj.full_path_grad(x)
        
        while self.objective_obj.objective_func(x - t * delta_x) > fx + self.alpha * t * np.dot(delta_fx, - delta_x):
            t = self.beta * t 

        return t

 
class BacktrackingLineSearchForProximal(LineSearch):
    """
    具象クラス : 近接勾配法のためのバックトラッキング

    Attributes:
        後で書く

    """

    def __init__(self, objective_obj, prox_operator_obj, beta = 0.7):
        """
        コンストラクタ
        Args:
            objective_obj : class[ObjectiveFunction の具象クラス]
                目的関数の正則化項を取り除いた部分の抽象クラスの具象クラスのインスタンス
                目的関数そのものではないことに注意！
            prox_operator_obj : class[ProximalOperator の具象クラス]
                近接作用素の抽象クラスの具象クラスのインスタンス            
            beta  : バックトラッキングのパラメータ 通常は 0.1 - 0.8 から選択

        """ 
        self. objective_obj =  objective_obj
        self.prox_operator_obj = prox_operator_obj
        self.beta = beta


    def line_search(self, x, delta_x = 0):        
        t = 1
        gx = self.objective_obj.objective_func(x)
        delta_g = self.objective_obj.full_path_grad(x)

        while 1:            
            gd = x - t * delta_g
            capital_g = ( x - self.prox_operator_obj.proximal_mapping(gd, t) ) / t

            left = self.objective_obj.objective_func( x - t * capital_g )
            right = gx - t * np.dot(delta_g, capital_g) + ( 0.5 * t) * np.dot(capital_g, capital_g)

            if left <= right:
                break
         
            t = self.beta * t
        
        return t


# class BacktrackingLineSearchForAcceleratedProximal(LineSearch):
#     """
#     具象クラス : 加速付き近接勾配法のためのバックトラッキング

#     Attributes:
#         後で書く

#     """

#     def __init__(self, objective_obj, prox_operator_obj, beta = 0.7):
#         """
#         コンストラクタ
#         Args:
#             objective_obj : class[ObjectiveFunction の具象クラス]
#                 目的関数の正則化項を取り除いた部分の抽象クラスの具象クラスのインスタンス
#                 目的関数そのものではないことに注意 ！
#             prox_operator_obj : class[ProximalOperator の具象クラス]
#                 近接作用素の抽象クラスの具象クラスのインスタンス            
#             beta  : バックトラッキングのパラメータ 通常は 0.1 - 0.8 から選択

#         """ 
#         self. objective_obj =  objective_obj
#         self.prox_operator_obj = prox_operator_obj
#         self.beta = beta


#     def line_search(self, x, y, delta_x = 0):        
#         t = 1
#         gy = self.objective_obj.objective_func(y)
#         delta_g = self.objective_obj.full_path_grad(y)

#         while 1:            
#             gd = y - t * delta_g
#             capital_g = ( y - self.prox_operator_obj.proximal_mapping(gd, t) ) / t
            
#             left = self.objective_obj.objective_func( y - t * capital_g )
#             right = gy - t * np.dot(delta_g, capital_g) + ( 0.5 * t) * np.dot(capital_g, capital_g)

#             if left <= right:
#                 break
         
#             t = self.beta * t
        
#         return t


if __name__ == "__main__":
    pass