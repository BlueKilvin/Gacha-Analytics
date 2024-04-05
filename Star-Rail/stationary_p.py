from Star-Rail import PITY_5STAR, PITY_4STAR, PITY_W5STAR, PITY_W4STAR
from stationary_distribution_method import PriorityPitySystem

# Call the preset tool to calculate the probability in the case of five-star and four-star
common_gacha_system = PriorityPitySystem([PITY_5STAR, PITY_4STAR, [0, 1]])
print('Permanent character pool probability', common_gacha_system.get_stationary_p())
light_cone_gacha_system = PriorityPitySystem([PITY_W5STAR, PITY_W4STAR, [0, 1]])
print('light cone probability',light_cone_gacha_system.get_stationary_p())