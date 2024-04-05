import numpy as np
from scipy.special import comb
from distribution_1d import *
from basic_models import PityModel


def calc_stationary_distribution(M):
    '''
    |1 0.5|   |x|
    |0 0.5|   |y|
    '''
    matrix_shape = np.shape(M)
    if matrix_shape[0] == matrix_shape[1]:
        pass
    else:
        print("Error: Imput should be square matrix.")
        return
    # subtract the diagonal matrix
    C = M - np.identity(matrix_shape[0])
    # The last row is set to 1
    C[matrix_shape[0]-1] = 1
    X = np.zeros(matrix_shape[0], dtype=float)
    X[matrix_shape[0]-1] = 1
    # Linear
    ans = np.linalg.solve(C, X)
    return ans

def multi_item_rarity(pity_p: list, once_pull_times: int, is_complete=True):
    P_m = np.zeros(once_pull_times+1, dtype=float)
    P_m[0] = 1
    for i in range(1, once_pull_times+1):
        if i < len(pity_p):
            P_m[i] = P_m[i-1] * (1-pity_p[i])
        else:
            P_m[i] = 0

    def build_n_time_matrix(pity_p, item_model, once_pull_times):
        '''
        TODO: build n time matrix, 7
        '''
        M = np.zeros((len(pity_p)-1, len(pity_p)-1), dtype=np.double)
        base_p = pity_p[1]
        for i in range(len(pity_p)-1):
            dist = item_model(1, item_pity=i).dist[:once_pull_times+1]
            dist = pad_zero(dist, once_pull_times+1)
            for j in range(1, once_pull_times+1):
                p_j = dist[j]
                M[once_pull_times-j][i] += p_j * (1-base_p) ** (once_pull_times-j)
                for k in range(j+1, once_pull_times+1):
                    M[once_pull_times-k][i] += p_j * base_p * (1-base_p) ** (once_pull_times-k) 
            if once_pull_times+i < len(pity_p)-1:
                M[once_pull_times+i][i] = 1 - sum(dist)
        print(M)
        print(M.sum(axis=0))
        return M

    def build_n_time_matrix_complete(pity_p, item_model, once_pull_times):
        '''
        TODO Verify the correctness of this function
        TODO Consider the situation where the guaranteed position is also less than the number of consecutive draws, 
            and check the correctness in this case.
        TODO 9,8
        '''

        matrix_size = max(len(pity_p)-1, once_pull_times)

        M = np.zeros((matrix_size, matrix_size), dtype=np.double)
        base_p = pity_p[1]
        for i in range(len(pity_p)-1):
            dists = item_model(once_pull_times, item_pity=i, multi_dist=True)
            for j in range(len(dists)):
                dists[j] = pad_zero(dists[j].dist[:once_pull_times+1], once_pull_times+1)
            for j in range(1, once_pull_times+1):
                for k in range(j, once_pull_times+1):
                    p_k = dists[j][k] * P_m[once_pull_times-k]
                    M[once_pull_times-k][i] += p_k
            if once_pull_times+i < len(pity_p)-1:
                M[once_pull_times+i][i] = 1 - sum(dists[1])
        return M
    
    item_model = PityModel(pity_p)

    if is_complete:
        M = build_n_time_matrix_complete(pity_p, item_model, once_pull_times)
    else:
        M = build_n_time_matrix(pity_p, item_model, once_pull_times)
    stationary_left = calc_stationary_distribution(M)
    
    '''
    first_item_p = np.zeros(once_pull_times+1, dtype=np.double)
    for i in range(len(pity_p)-1):
        dist = pad_zero(item_model(1, item_pity=i).dist[:once_pull_times+1], once_pull_times+1)
        first_item_p += stationary_left[i] * dist
    
    ans = np.zeros(once_pull_times+1, dtype=np.double)
    for i in range(1, once_pull_times+1):
        for j in range(1, once_pull_times+2-i):
            ans[i] += first_item_p[j] * comb(once_pull_times-j, i-1) * pity_p[1] ** (i-1) * (1-pity_p[1]) ** (once_pull_times-j-i+1)

    ####### Check this against the new pity pull #########
    '''
    # Uses the generalized method to calculate the probability of drawing multiple props in a row
    ans = np.zeros(once_pull_times+1, dtype=np.double)
    for i in range(1, once_pull_times+1):
        for j in range(len(pity_p)-1):
            dist = item_model(i, item_pity=j)
            for k in range(1, len(dist.dist[:once_pull_times+1])):
                ans[i] += stationary_left[j] * dist.dist[k] * P_m[once_pull_times-k]
    ans[0] = 1 - sum(ans[1:])
    return ans


class PriorityPitySystem(object):
    def __init__(self, item_p_list: list, extra_state = 1, remove_pity = False) -> None:
        # TODO extra state 
        self.item_p_list = item_p_list  
        self.item_types = len(item_p_list)  # How many props in total?
        self.remove_pity = remove_pity
        self.extra_state = extra_state  
        
        self.pity_state_list = []  # Record how many states each prop retains
        self.pity_pos_max = []  
        for pity_p in item_p_list:
            self.pity_state_list.append(len(pity_p)+extra_state-1)
            self.pity_pos_max.append(len(pity_p)-1)

        self.max_state = 1 
        for pity_state in self.pity_state_list:
            self.max_state = self.max_state * pity_state

        self.transfer_matrix = self.get_transfer_matrix()  
        self.stationary_distribution = calc_stationary_distribution(self.transfer_matrix)

    def item_pity_p(self, item_type, p_pos):
        return self.item_p_list[item_type][min(p_pos, self.pity_pos_max[item_type])]

    def get_state(self, state_num) -> list:
        pity_state = []
        for i in self.pity_state_list[::-1]:
            pity_state.append(state_num % i)
            state_num = state_num // i
        return pity_state[::-1]

    def get_number(self, pity_state) -> int:
        number = 0
        last = 1
        for i, s in zip(self.pity_state_list[::-1], pity_state[::-1]):
            number += s * last
            last *= i
        return number

    def get_next_state(self, pity_state, get_item=None) -> list:
        # Pity state check
        next_state = []
        for i in range(self.item_types):
            if get_item == i:  
                next_state.append(0)
            else:  # No such items were obtained
                if self.remove_pity and get_item is not None:
                    if i > get_item:
                        next_state.append(0)
                        continue
                next_state.append(min(self.pity_state_list[i]-1, pity_state[i]+1))
        return next_state

    def get_transfer_matrix(self) -> np.ndarray:
        M = np.zeros((self.max_state, self.max_state)) 

        for i in range(self.max_state):
            left_p = 1
            current_state = self.get_state(i)
            for item_type, p_pos in zip(range(self.item_types), current_state):
                next_state = self.get_next_state(current_state, item_type)
                transfer_p = min(left_p, self.item_pity_p(item_type, p_pos+1))
                M[self.get_number(next_state)][i] = transfer_p
                left_p = left_p - transfer_p
            next_state = self.get_next_state(current_state, None)
            M[self.get_number(next_state)][i] = left_p
        return  M
    
    def get_stationary_p(self) -> list:
        stationary_p = np.zeros(len(self.item_p_list))
        for i in range(self.max_state):
            current_state = self.get_state(i)
            for j, item_state in enumerate(current_state):
                if item_state == 0:
                    stationary_p[j] += self.stationary_distribution[i]
                    break
        return stationary_p

    def get_type_distribution(self, type) -> np.ndarray:
        ans = np.zeros(self.pity_state_list[type]+1)
        # print('shape', ans.shape)
        for i in range(self.max_state):
            left_p = 1
            current_state = self.get_state(i)
            for item_type, p_pos in zip(range(type), current_state[:type]):
                left_p -= self.item_pity_p(item_type, p_pos+1)
            transfer_p = min(max(0, left_p), self.item_pity_p(type, current_state[type]+1))
            next_pos = min(self.pity_state_list[type], current_state[type]+1)
            ans[next_pos] += self.stationary_distribution[i] * transfer_p
        return ans/sum(ans)