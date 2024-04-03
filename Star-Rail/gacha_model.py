'''
    Note that the probability model used in this module is only speculation and cannot be guaranteed to be completely accurate.
'''
from GGanalysis.distribution_1d import *
from GGanalysis.gacha_layers import *
from GGanalysis.basic_models import *

__all__ = [
    'PITY_5STAR',
    'PITY_4STAR',
    'PITY_W5STAR',
    'PITY_W4STAR',
    'common_5star',
    'common_4star',
    'up_5star_character',
    'up_4star_character',
    'up_4star_specific_character',
    'common_5star_weapon',
    'common_4star_weapon',
    'up_5star_weapon',
    'up_4star_weapon',
    'up_4star_specific_weapon',
]


PITY_5STAR = np.zeros(91)
PITY_5STAR[1:74] = 0.006
PITY_5STAR[74:90] = np.arange(1, 17) * 0.06 + 0.006
PITY_5STAR[90] = 1

PITY_4STAR = np.zeros(11)
PITY_4STAR[1:9] = 0.051
PITY_4STAR[9] = 0.051 + 0.51
PITY_4STAR[10] = 1

PITY_W5STAR = np.zeros(81)
PITY_W5STAR[1:66] = 0.008
PITY_W5STAR[66:80] = np.arange(1, 15) * 0.07 + 0.008
PITY_W5STAR[80] = 1

PITY_W4STAR = np.array([0,0.066,0.066,0.066,0.066,0.066,0.066,0.066,0.066*7.1,0.066*13.1,1])


common_5star = PityModel(PITY_5STAR)
common_4star = PityModel(PITY_4STAR)

up_5star_character = DualPityModel(PITY_5STAR, [0, 0.5, 1])
up_4star_character = DualPityModel(PITY_4STAR, [0, 0.5, 1])
up_4star_specific_character = DualPityBernoulliModel(PITY_4STAR, [0, 0.5, 1], 1/3)

common_5star_weapon = PityModel(PITY_W5STAR)
common_4star_weapon = PityModel(PITY_W4STAR)
up_5star_weapon = DualPityModel(PITY_W5STAR, [0, 0.75, 1])
up_4star_weapon = DualPityModel(PITY_W4STAR, [0, 0.75, 1])
up_4star_specific_weapon = DualPityBernoulliModel(PITY_W4STAR, [0, 0.75, 1], 1/3)


if __name__ == '__main__':
    # print(PITY_W5STAR)
    # Calculate quantiles
    quantile_pos = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    print("Select quantiles"+str(quantile_pos))
    for c in range(0, 7):
        for w in range(0, 6):
            dist = up_5star_character(c) * up_5star_weapon(w)
            print(f"{c}soul{w}stack "+str(dist.quantile_point(quantile_pos)))