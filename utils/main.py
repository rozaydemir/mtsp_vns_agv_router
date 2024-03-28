from pdptw import PDPTW
from alns import ALNS

data = "Instances/lrc104.txt" # datayı yükle
problem = PDPTW.readInstance(data)

# Static parameters
nDestroyOps = 6  #number of destroy operations, çeşitlilik sağlanmak istenirse 9 a çıkar
nRepairOps = 2  # number of repair operations # çeşitlilik sağlanmak istenirse 3 e çıkar
minSizeNBH = 1  #Minimum size of neighborhood
nIterations = 100  #Algoritma 100 kez tekrarlanacak(100 kez destroy ve rerair işlemlerini tekrarlayacak)

# Parameters to tune:
maxPercentageNHB = 5  #Maximum Percentage for Neighborhood
decayParameter = 0.15
noise = 0.015  #gürültü ekleme, çözüm uzayında daha çeşitli noktaları keşfetmeye yardımcı olur.

alns = ALNS(problem, nDestroyOps, nRepairOps, nIterations, minSizeNBH, maxPercentageNHB, decayParameter, noise)
alns.execute()
