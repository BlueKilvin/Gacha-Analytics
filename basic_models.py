from distribution_1d import *
#from gacha_layers import *
from recursion_methods import GeneralCouponCollection
from typing import Union

class GachaModel(object):
    pass

class CommonGachaModel(GachaModel):
    def __init__(self) -> None:
        super().__init__()
        self.layers = []
    
    def __call__(self, item_num: int=1, multi_dist: bool=False, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        parameter_list = self._build_parameter_list(*args, **kwds)
        # Build Parameter List 
        if args != () and kwds != {} and parameter_list is None:
            raise Exception('Parameters is not defined.')

        if item_num == 0:
            return FiniteDist([1])
        
        if multi_dist:
            return self._get_multi_dist(item_num, parameter_list)
        
        return self._get_dist(item_num, parameter_list)

    
    def _build_parameter_list(self, *args: any, **kwds: any) -> list:
        parameter_list = []
        for i in range(len(self.layers)):
            parameter_list.append([[], {}])
        return parameter_list

    
    def _get_multi_dist(self, end_pos: int, parameter_list: list=None):
        input_dist = self._forward(parameter_list)
        ans_list = [FiniteDist([1]), input_dist[1]]
        for i in range(1, end_pos):
            
            ans_list.append(ans_list[i] * input_dist[0])
            ans_list[i+1].exp = input_dist[1].exp + input_dist[0].exp * i
            ans_list[i+1].var = input_dist[1].var + input_dist[0].var * i
        return ans_list

    
    def _get_dist(self, item_num: int, parameter_list: list=None):
        ans_dist = self._forward(parameter_list)
        ans: FiniteDist = ans_dist[1] * ans_dist[0] ** (item_num - 1)
        ans.exp = ans_dist[1].exp + ans_dist[0].exp * (item_num - 1)
        ans.var = ans_dist[1].var + ans_dist[0].var * (item_num - 1)
        return ans

    
    def _forward(self, parameter_list: list=None):
        ans_dist = None
        
        if parameter_list is None:
            for layer in self.layers:
                ans_dist = layer(ans_dist)
            return ans_dist
        
        for parameter, layer in zip(parameter_list, self.layers):
            ans_dist = layer(ans_dist, *parameter[0], **parameter[1])
        return ans_dist

class BernoulliGachaModel(GachaModel):
    def __init__(self, p, e_error = 1e-8, max_dist_len=1e5) -> None:
        super().__init__()
        self.p = p  
        self.e_error = e_error
        self.max_dist_len = max_dist_len

    def __call__(self, item_num: int, calc_pull: int=None) -> FiniteDist:
        # Returns the distribution of the number of items drawn
        output_E = item_num / self.p
        output_D = item_num * (1 - self.p) / self.p ** 2
        if calc_pull is None:
            test_len = max(int(output_E), 2)
            while True:
                x = np.arange(test_len+1)
                output_dist = self.p * (binom.pmf(item_num-1, x-1, self.p))
                output_dist[0] = 0
                output_dist = FiniteDist(output_dist)
                calc_error = abs(calc_expectation(output_dist)-output_E)/output_E
                if calc_error < self.e_error or test_len > self.max_dist_len:
                    if test_len > self.max_dist_len:
                        print('Warning: distribution is too long! len:', test_len, 'Error:', calc_error)
                    output_dist.exp = output_E
                    output_dist.var = output_D
                    return output_dist
                test_len *= 2

        x = np.arange(calc_pull+1)
        output_dist = self.p * (binom.pmf(item_num-1, x-1, self.p))
        output_dist[0] = 0
        output_dist = FiniteDist(output_dist)
        return output_dist

    # The probability of drawing item_num props in the xth draw
    '''
    def _get_dist(self, item_num, pulls):  
        x = np.arange(pulls+1)
        dist = self.p * (binom.pmf(0, x-1, self.p))
        dist[0] = 0
        return finite_dist_1D(dist)
    '''

class CouponCollectorModel(CommonGachaModel):
    #Drawing cards props
    def __init__(self, item_types, e_error = 1e-6, max_dist_len = 1e5) -> None:
        super().__init__()
        self.layers.append(CouponCollectorLayer(item_types, None, e_error, max_dist_len))
    
    def __call__(self, initial_types: int = 0, target_types: int = None, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(1, False, initial_types, target_types, *args, **kwds)

    def _build_parameter_list(self, initial_types: int = 0, target_types: int = None) -> list:
        parameter_list = [
            [[], {'initial_types':initial_types, 'target_types':target_types}],
        ]
        return parameter_list
    
class PityCouponCollectorModel(CommonGachaModel):
    def __init__(self, pity_p, item_types, e_error = 1e-6, max_dist_len = 1e5) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p))
        self.layers.append(CouponCollectorLayer(item_types, None, e_error, max_dist_len))
    
    def __call__(self, initial_types: int = 0, item_pity = 0, target_types: int = None, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(1, False, item_pity, initial_types, target_types, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0, initial_types: int = 0, target_types: int = None) -> list:
        parameter_list = [
            [[], {'item_pity':item_pity}],
            [[], {'initial_types':initial_types, 'target_types':target_types}],
        ]
        return parameter_list
    
class DualPityCouponCollectorModel(CommonGachaModel):
    def __init__(self, pity_p1, pity_p2, item_types, e_error = 1e-6, max_dist_len = 1e5) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p1))
        self.layers.append(PityLayer(pity_p2))
        self.layers.append(CouponCollectorLayer(item_types, None, e_error, max_dist_len))
    
    def __call__(self, initial_types: int = 0, item_pity = 0, up_pity: int = 0, target_types: int = None, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(1, False, item_pity, up_pity, initial_types, target_types, *args, **kwds)

    def _build_parameter_list(self, item_pity: int = 0, up_pity:int = 0, initial_types: int = 0, target_types: int = None) -> list:
        parameter_list = [
            [[], {'item_pity':item_pity}],
            [[], {'item_pity':up_pity}],
            [[], {'initial_types':initial_types, 'target_types':target_types}],
        ]
        return parameter_list

class GeneralCouponCollectorModel(GachaModel):
    def __init__(self, p_list: Union[list, np.ndarray], item_name: list[str]=None, e_error = 1e-6, max_dist_len = 1e5) -> None:
        super().__init__()
        self.e_error = e_error
        self.max_dist_len = max_dist_len
        self.model = GeneralCouponCollection(p_list, item_name)

    def __call__(self, init_item: list=None, target_item: list=None) -> FiniteDist:
        # 输入处理
        if init_item is None:
            init_state = self.model.default_init_state
        else:
            init_state = self.model.encode_state_number(init_item)
        if target_item is None:
            target_state = self.model.default_target_state
        else:
            target_state = self.model.encode_state_number(target_item)
        output_E = self.model.get_expectation(init_state, target_state)
        test_len = max(int(output_E), 2)
        while True:
            output_dist = cdf2dist(self.model.get_collection_p(test_len, init_state, target_state))
            calc_error = abs(calc_expectation(output_dist)-output_E)/output_E
            if calc_error < self.e_error or test_len > self.max_dist_len:
                if test_len > self.max_dist_len:
                    print('Warning: distribution is too long! len:', test_len, 'Error:', calc_error)
                output_dist.exp = output_E
                return output_dist
            test_len *= 2

class PityModel(CommonGachaModel):
    def __init__(self, pity_p) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p))
    
    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity = 0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0) -> list:
        parameter_list = [[[], {'item_pity':item_pity}]]
        return parameter_list

class DualPityModel(CommonGachaModel):
    def __init__(self, pity_p1, pity_p2) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p1))
        self.layers.append(PityLayer(pity_p2))

    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity = 0, up_pity = 0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, up_pity, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0, up_pity: int=0) -> list:
        parameter_list = [
            [[], {'item_pity':item_pity}],
            [[], {'item_pity':up_pity}],
        ]
        return parameter_list

class PityBernoulliModel(CommonGachaModel):
    def __init__(self, pity_p, p, e_error = 1e-8, max_dist_len=1e5) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p))
        self.layers.append(BernoulliLayer(p, e_error, max_dist_len))
    
    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity=0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0) -> list:
        parameter_list = [
            [[], {'item_pity':item_pity}],
            [[], {}],
        ]
        return parameter_list

class DualPityBernoulliModel(CommonGachaModel):
    def __init__(self, pity_p1, pity_p2, p, e_error = 1e-8, max_dist_len=1e5) -> None:
        super().__init__()
        self.layers.append(PityLayer(pity_p1))
        self.layers.append(PityLayer(pity_p2))
        self.layers.append(BernoulliLayer(p, e_error, max_dist_len))
    
    def __call__(self, item_num: int = 1, multi_dist: bool = False, item_pity = 0, up_pity = 0, *args: any, **kwds: any) -> Union[FiniteDist, list]:
        return super().__call__(item_num, multi_dist, item_pity, up_pity, *args, **kwds)

    def _build_parameter_list(self, item_pity: int=0, up_pity: int=0) -> list:
        parameter_list = [
            [[], {'item_pity':item_pity}],
            [[], {'item_pity':up_pity}],
            [[], {}],
        ]
        return parameter_list