from typing import Union
import numpy as np
import math
from scipy.signal import convolve

def linear_p_increase(base_p=0.01, pity_begin=100, step=1, hard_pity=100):

    ans = np.zeros(hard_pity+1)
    ans[1:pity_begin] = base_p
    ans[pity_begin:hard_pity+1] = np.arange(1, hard_pity-pity_begin+2) * step + base_p
    return np.minimum(ans, 1)

def calc_expectation(dist: Union['FiniteDist', list, np.ndarray]) -> float:

    if isinstance(dist, FiniteDist):
        dist = dist.dist
    else:
        dist = np.array(dist)
    return sum(np.arange(len(dist)) * dist)

def calc_variance(dist: Union['FiniteDist', list, np.ndarray]) -> float:

    if isinstance(dist, FiniteDist):
        dist = dist.dist
    else:
        dist = np.array(dist)
    use_pulls = np.arange(len(dist))
    exp = sum(use_pulls * dist)
    return sum((use_pulls - exp) ** 2 * dist)

def dist_squeeze(dist: Union['FiniteDist', list, np.ndarray], squeeze_factor) -> 'FiniteDist':

    n = math.ceil((len(dist)-1)/squeeze_factor)+1
    new_arr = np.zeros(n, dtype=float)
    new_arr[0] = dist[0]
    for i in range(1, n):
        new_arr[i] = np.sum(dist[(i-1)*squeeze_factor+1:i*squeeze_factor+1])
    return FiniteDist(new_arr)

def dist2cdf(dist: Union[np.ndarray, 'FiniteDist']) -> np.ndarray:

    if isinstance(dist, FiniteDist):
        return np.cumsum(dist.dist)
    return np.cumsum(dist)

def cdf2dist(cdf: np.ndarray) -> 'FiniteDist':

    if len(cdf) == 1:
        return FiniteDist([1])
    pdf = np.array(cdf)
    pdf[1:] -= pdf[:-1].copy()
    return FiniteDist(pdf)

def p2dist(pity_p: Union[list, np.ndarray]) -> 'FiniteDist':
    # Enter the guaranteed parameter list
    # the probability of position 0 should be 0
    temp = 1
    dist = [0]
    for i in range(1, len(pity_p)):
        dist.append(temp * pity_p[i])
        temp *= (1-pity_p[i])
    return FiniteDist(dist)

def p2exp(pity_p: Union[list, np.ndarray]):
    return calc_expectation(p2dist(pity_p))

def p2var(pity_p: Union[list, np.ndarray]):
    return calc_variance(p2dist(pity_p))

def table2matrix(state_num, state_trans):
    '''
    Convert list to matrix

    Epitomized Path & Fate Points
    state_num = {'get':0, 'fate1UP':1, 'fate1':2, 'fate2':3}
    state_trans = [
        ['get', 'get', 0.375],
        ['get', 'fate1UP', 0.375],
        ['get', 'fate1', 0.25],
        ['fate1UP', 'get', 0.375],
        ['fate1UP', 'fate2', 1-0.375],
        ['fate1', 'get', 0.5],
        ['fate1', 'fate2', 0.5],
        ['fate2', 'get', 1]
    ]
    '''
    M = np.zeros((len(state_num), len(state_num)))
    for name_a, name_b, p in state_trans:
        a = state_num[name_a]
        b = state_num[name_b]
        M[b][a] = p
    # Check whether the sum of the exit probabilities of each node is 1
    '''
    for index, element in enumerate(np.sum(M, axis=0)):
        if element != 1:
            raise Warning('The sum of probabilities is not 1 at position '+str(index))
    '''
    return M

def pad_zero(dist:np.ndarray, target_len):
    if target_len <= len(dist):
        return dist
    return np.pad(dist, (0, target_len-len(dist)), 'constant', constant_values=0)

def cut_dist(dist: Union[np.ndarray, 'FiniteDist'], cut_pos):
    if cut_pos == 0:
        return dist
    # Normalized
    ans = dist[cut_pos:].copy()
    ans[0] = 0
    return ans/sum(ans)

class FiniteDist(object):  
    def __init__(self, dist: Union[list, np.ndarray, 'FiniteDist'] = [1]) -> None:
        if isinstance(dist, FiniteDist):
            self.dist = np.array(dist.dist, dtype=float)
            return
        if len(np.shape(dist)) > 1:
            raise Exception('Not 1D distribution.')
        self.dist = np.array(dist, dtype=float)  # numpy.ndarray
        if len(self.dist) == 0:
            self.dist = np.zeros(1, dtype=float)

    def __getattr__(self, key):  
        # Basic Specs
        if key in ['exp', 'var', 'p_sum']:
            self.calc_dist_attribution()
            if key == 'exp':
                return self.exp
            if key == 'var':
                return self.var
            if key == 'p_sum':
                return self.p_sum
        # cumulative probability 
        if key == 'cdf':
            self.calc_cdf()
            return self.cdf
        # ---------------------
        if key in ['entropy_rate', 'randomness_rate']:
            self.calc_entropy_attribution()
            if key == 'entropy_rate':
                return self.entropy_rate
            if key == 'randomness_rate':
                return self.randomness_rate
    
    def __iter__(self): 
        return iter(self.dist)
    def __setitem__(self, sliced, value: Union[int, float, np.ndarray]) -> None:
        self.dist[sliced] = value
    def __getitem__(self, sliced):
        return self.dist[sliced]

    def calc_cdf(self):
        self.cdf = dist2cdf(self.dist)

    def calc_dist_attribution(self, p_error=1e-6) -> None:
        self.p_sum = sum(self.dist)
        if abs(self.p_sum-1) > p_error: 
            self.exp = float('nan')
            self.var = float('nan')
            return
        use_pulls = np.arange(self.__len__())
        self.exp = sum(use_pulls * self.dist)
        self.var = sum((use_pulls-self.exp) ** 2 * self.dist)

    def calc_entropy_attribution(self, p_error=1e-6) -> None:
        if abs(self.p_sum-1) > p_error:
            self.entropy_rate = float('nan')
            self.randomness_rate = float('nan')
            return
        # ====================NO ZERO===================
        temp = np.zeros(len(self.dist))
        temp[0] = 1
        self.entropy_rate = -sum(self.dist * np.log2(self.dist+temp)) / self.exp
        self.randomness_rate = self.entropy_rate / (-1/self.exp * np.log2(1/self.exp) - (1-1/self.exp) * np.log2(1-1/self.exp))

    def quantile_point(self, quantile_p):
        return np.searchsorted(self.cdf, quantile_p, side='left')

    def p_normalization(self) -> None:
        self.dist = self.dist/sum(self.dist)
        self.calc_dist_attribution()

    def __add__(self, other: 'FiniteDist') -> 'FiniteDist':
        target_len = max(len(self), len(other))
        return FiniteDist(pad_zero(self.dist, target_len) + pad_zero(other.dist, target_len))

    def __mul__(self, other: Union['FiniteDist', float, int, np.float64, np.int32]) -> 'FiniteDist':
        # TODO 1,5
        if isinstance(other, FiniteDist):
            return FiniteDist(convolve(self.dist, other.dist))
        else:
            return FiniteDist(self.dist * other)
    def __rmul__(self, other: Union['FiniteDist', float, int, np.float64, np.int32]) -> 'FiniteDist':
        return self * other

    def __truediv__(self, other: Union[float, int]) -> 'FiniteDist':
        return FiniteDist(self.dist / other)
    def __pow__(self, pow_times: int) -> 'FiniteDist':
        ans = np.ones(1)
        if pow_times == 0:
            return FiniteDist(ans)
        if pow_times == 1:
            return self
        t = pow_times
        temp = self.dist
        while True:
            if t % 2:
                ans = convolve(ans, temp)
            t = t//2
            if t == 0:
                break
            temp = convolve(temp, temp)
        return FiniteDist(ans)

    def __str__(self) -> str:
        return f"finite 1D dist {self.dist}"

    def __len__(self) -> int:
        return len(self.dist)

if __name__ == "__main__":
    pass