from copy import deepcopy
from functools import lru_cache
from itertools import permutations
from typing import Callable

import numpy as np

#from games.honkai_star_rail.relic_data import *
#from scored_item import ScoredItem, ScoredItemSet
#Import these from external hook

"""
    Honkai Impact: Star Rail Relic Types
"""
__all__ = [
    "StarRailRelic",
    "StarRailRelicSet",
    "StarRailCavernRelics",
    "StarRailPlanarOrnaments",
    "RELIC_TYPES",
    "CAVERN_RELICS",
    "PLANAR_ORNAMENTS",
    "STAT_NAME",
    "W_MAIN_STAT",
    "W_SUB_STAT",
    "DEFAULT_MAIN_STAT",
    "DEFAULT_STAT_SCORE",
    "DEFAULT_STAT_COLOR",
]

MINOR_RANKS = [8, 9, 10]
RANK_MULTI = 1
STATS_WEIGHTS = {}


def set_using_weight(new_weight: dict):
    global STATS_WEIGHTS
    STATS_WEIGHTS = new_weight
    # print('Refresh weight cache!')
    get_init_state.cache_clear()
    get_state_level_up.cache_clear()

def dict_weight_sum(weights: dict):
    return sum(weights.values())

def get_combinations_p(stats_p: dict, select_num=4):
    ans = {}
    weight_all = dict_weight_sum(stats_p)
    for perm in permutations(list(stats_p.keys()), select_num):
        p, s = 1, weight_all
        for m in perm:
            p *= stats_p[m] / s
            s -= stats_p[m]
        
        perm_key = tuple(sorted(perm))
        if perm_key in ans:
            ans[perm_key] += p
        else:
            ans[perm_key] = p
    return ans

@lru_cache(maxsize=65536)
def get_init_state(stat_comb, default_weight=0) -> ScoredItem:
    #Obtain the initial distribution of the five-star relic 
    score_dist = np.zeros(40 * RANK_MULTI + 1)
    sub_stat_exp = {}
    for m in stat_comb:
        sub_stat_exp[m] = np.zeros(40 * RANK_MULTI + 1)

    for i in range(8, 11):
        s1 = STATS_WEIGHTS.get(stat_comb[0], default_weight) * i * RANK_MULTI
        for j in range(8, 11):
            s2 = s1 + STATS_WEIGHTS.get(stat_comb[1], default_weight) * j * RANK_MULTI
            for k in range(8, 11):
                s3 = (
                    s2
                    + STATS_WEIGHTS.get(stat_comb[2], default_weight) * k * RANK_MULTI
                )
                for l in range(8, 11):
                    # s4 
                    s4 = (
                        s3
                        + STATS_WEIGHTS.get(stat_comb[3], default_weight)
                        * l
                        * RANK_MULTI
                    )
                    
                    L = int(s4)
                    R = L + 1
                    w_L = R - s4
                    w_R = s4 - L
                    R = min(R, 40 * RANK_MULTI)
                    
                    score_dist[L] += w_L
                    sub_stat_exp[stat_comb[0]][L] += i * w_L
                    sub_stat_exp[stat_comb[1]][L] += j * w_L
                    sub_stat_exp[stat_comb[2]][L] += k * w_L
                    sub_stat_exp[stat_comb[3]][L] += l * w_L
                    score_dist[R] += w_R
                    sub_stat_exp[stat_comb[0]][R] += i * w_R
                    sub_stat_exp[stat_comb[1]][R] += j * w_R
                    sub_stat_exp[stat_comb[2]][R] += k * w_R
                    sub_stat_exp[stat_comb[3]][R] += l * w_R
    
    for m in stat_comb:
        sub_stat_exp[m] = np.divide(
            sub_stat_exp[m],
            score_dist,
            out=np.zeros_like(sub_stat_exp[m]),
            where=score_dist != 0,
        )
        sub_stat_exp[m] = np.trim_zeros(sub_stat_exp[m], "b")
    score_dist /= 81
    score_dist = np.trim_zeros(score_dist, "b")
    return ScoredItem(score_dist, sub_stat_exp)

@lru_cache(maxsize=65536)
def get_state_level_up(stat_comb, default_weight=0) -> ScoredItem:
   
    score_dist = np.zeros(10 * RANK_MULTI + 1)
    sub_stat_exp = {}
    for m in stat_comb:
        sub_stat_exp[m] = np.zeros(10 * RANK_MULTI + 1)
   
    for stat in stat_comb:
        for j in range(8, 11):
            score = STATS_WEIGHTS.get(stat, default_weight) * j * RANK_MULTI
            
            L = int(score)
            R = L + 1
            w_L = R - score
            w_R = score - L
            R = min(R, 10 * RANK_MULTI)
            
            score_dist[L] += w_L
            sub_stat_exp[stat][L] += j * w_L
            score_dist[R] += w_R
            sub_stat_exp[stat][R] += j * w_R
    
    for m in stat_comb:
        sub_stat_exp[m] = np.divide(
            sub_stat_exp[m],
            score_dist,
            out=np.zeros_like(sub_stat_exp[m]),
            where=score_dist != 0,
        )
        sub_stat_exp[m] = np.trim_zeros(sub_stat_exp[m], "b")
    score_dist /= 12
    score_dist = np.trim_zeros(score_dist, "b")
    return ScoredItem(score_dist, sub_stat_exp)

class StarRailRelic(ScoredItem):
    

    def __init__(
        self,
        type: str = "hands",  
        type_p = 1/4, 
        main_stat: str = None,  
        sub_stats_select_weight: dict = W_SUB_STAT, 
        stats_score: dict = DEFAULT_STAT_SCORE,  
        p_4sub: float = P_INIT4_DROP,  
        sub_stats_filter: Callable[..., bool] = None, 
        forced_combinations = None, 
    ) -> None:
        
        self.type = type
        if main_stat is not None:
            self.main_stat = main_stat
        else:
            self.main_stat = DEFAULT_MAIN_STAT[self.type]
        drop_p = (
            type_p
            * W_MAIN_STAT[self.type][self.main_stat]
            / dict_weight_sum(W_MAIN_STAT[self.type])
        )
        
        self.sub_stats_weight = deepcopy(sub_stats_select_weight)
        if self.main_stat in self.sub_stats_weight:
            del self.sub_stats_weight[self.main_stat]
       
        self.p_4sub = p_4sub
        
        self.stats_score = stats_score
        if self.stats_score != STATS_WEIGHTS:
            set_using_weight(self.stats_score)
       
        ans = ScoredItem()
        if forced_combinations is None:
            self.sub_stats_combinations = get_combinations_p(
                stats_p=self.sub_stats_weight, select_num=4
            )
        else:
            self.sub_stats_combinations = forced_combinations
       
        for stat_comb in list(self.sub_stats_combinations.keys()):
            if sub_stats_filter is not None:
                if sub_stats_filter(stat_comb) is False:
                    continue
            temp_base = get_init_state(stat_comb)
            temp_level_up = get_state_level_up(stat_comb)
           
            temp_3 = (
                temp_base
                * temp_level_up
                * temp_level_up
                * temp_level_up
                * temp_level_up
            )
            temp_4 = temp_3 * temp_level_up
            ans += (
                (1 - self.p_4sub) * temp_3 + self.p_4sub * temp_4
            ) * self.sub_stats_combinations[stat_comb]
        super().__init__(ans.score_dist, ans.sub_stats_exp, drop_p, stats_score=self.stats_score)

class StarRailRelicSet(ScoredItemSet):
    def __init__(
        self,
        main_stat: dict = DEFAULT_MAIN_STAT,
        stats_score: dict = DEFAULT_STAT_SCORE,
        set_types: list = CAVERN_RELICS,
        p_4sub: float = P_INIT4_DROP,  
        type_p = 1/8, 
        sub_stats_filter: Callable[..., bool] = None, 
    ) -> None:
        self.main_stat = main_stat
        self.stats_score = stats_score
        self.p_4sub = p_4sub
        self.set_types = set_types
        item_set = {}
        for type in set_types:
            item_set[type] = StarRailRelic(
                type=type,
                type_p=type_p,
                main_stat=main_stat[type],
                stats_score=stats_score,
                p_4sub=self.p_4sub,
                sub_stats_filter=sub_stats_filter,
            )
        super().__init__(item_set)

class StarRailCavernRelics(StarRailRelicSet):
    def __init__(
        self,
        main_stat: dict = DEFAULT_MAIN_STAT,
        stats_score: dict = DEFAULT_STAT_SCORE,
        set_types: list = CAVERN_RELICS,
        p_4sub: float = P_INIT4_DROP,  
        type_p = 1/8,  
        sub_stats_filter: Callable[..., bool] = None, 
    ) -> None:
        super().__init__(
            main_stat = main_stat,
            stats_score = stats_score,
            set_types = set_types,
            p_4sub = p_4sub,
            type_p = type_p,
            sub_stats_filter = sub_stats_filter, 
        )

class StarRailPlanarOrnaments(StarRailRelicSet):
    def __init__(
        self,
        main_stat: dict = DEFAULT_MAIN_STAT,
        stats_score: dict = DEFAULT_STAT_SCORE,
        set_types: list = PLANAR_ORNAMENTS,
        p_4sub: float = P_INIT4_DROP, 
        type_p = 1/4,  
        sub_stats_filter: Callable[..., bool] = None, 
    ) -> None:
        super().__init__(
            main_stat = main_stat,
            stats_score = stats_score,
            set_types = set_types,
            p_4sub = p_4sub,
            type_p = type_p,
            sub_stats_filter = sub_stats_filter, 
        )

if __name__ == "__main__":
    pass