import random
import numpy as np

# Problem parametreleri
n_pickup = 5  # Pickup noktalarının sayısı
n_delivery = 5 # Delivery noktalarının sayısı
n_total_nodes = n_pickup + n_delivery  # Toplam düğüm sayısı (pickup ve delivery noktaları)
n_vehicles = 3  # Kullanılacak araç sayısı
vehicle_capacity = 10  # Araçların kapasitesi
max_iterations = 5000  # Genetik algoritma için maksimum iterasyon sayısı
population_size = 50  # Popülasyon boyutu, çözüm adaylarının sayısı

# Pickup ve delivery noktalarının koordinatlarını rastgele oluşturuyoruz
nodes = np.random.rand(n_total_nodes, 2) * 100  # Her düğümün rastgele bir koordinatı olacak

# Araçların depolarının başlangıç noktalarını tanımlıyoruz (depolar aynı noktada başlıyor)
depots = np.array([[0, 0] for _ in range(n_vehicles)])

# Öklid uzaklığını hesaplayan fonksiyon
def calculate_distance(p1, p2):
    # İki nokta arasındaki düz çizgi mesafesini hesaplar (Öklid mesafesi)
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Fitness fonksiyonu: her çözüm için uygunluk değerini hesaplar
# Fitness fonksiyonu, çözümün ne kadar iyi olduğunu değerlendirir.
# Burada uygunluk, toplam yol mesafesi ile ölçülüyor (yani düşük mesafe daha iyi çözüm demektir).
def fitness(solution):
    total_distance = 0  # Toplam mesafeyi tutacak değişken
    for vehicle in solution:
        # Her aracın rotasını oluşturuyoruz: depo -> düğümler -> depo
        route = [depots[0]] + [nodes[i] for i in vehicle] + [depots[0]]
        # Rotadaki her iki düğüm arasındaki mesafeyi topluyoruz
        for i in range(len(route) - 1):
            total_distance += calculate_distance(route[i], route[i + 1])
    return total_distance  # Toplam mesafe geri döndürülür, bu fitness değeridir

# Popülasyonu başlatma: İlk çözüm kümesini (popülasyonu) rastgele oluşturuyoruz
def initialize_population():
    population = []
    for _ in range(population_size):
        solution = []
        nodes_list = list(range(n_total_nodes))  # Düğümlerden bir liste oluştur
        random.shuffle(nodes_list)  # Düğüm listesini rastgele sırala
        # Rastgele sıralanan düğümleri her araç için bir rota olacak şekilde böl
        solution = [nodes_list[i::n_vehicles] for i in range(n_vehicles)]
        population.append(solution)  # Her çözüm popülasyona eklenir
    return population  # Rastgele popülasyon döndürülür

# Seçilim (Selection): Turnuva seçimi ile bireyleri seçiyoruz
# Bu aşamada en uygun çözümleri bulmak için bireyleri seçiyoruz.
# Daha iyi bireylerin daha yüksek seçilme şansı olur.
def selection(population):
    # Popülasyondan rastgele 3 birey seç ve onları uygunluklarına göre sırala
    selected = random.sample(population, 3)
    selected.sort(key=fitness)  # Uygunluk değerine göre sırala
    return selected[0]  # En iyi (en küçük mesafeye sahip) olanı geri döndür

# Çaprazlama (Crossover): İki ebeveynden yeni bireyler oluşturuyoruz
# Çaprazlama, iki çözümün genetik materyallerini karıştırarak yeni bireyler (çocuklar) üretir.
# Bu sayede popülasyona yeni ve daha iyi çözüm adayları eklenir.
def crossover(parent1, parent2):
    child1, child2 = [], []
    # Her araç için çaprazlama yapıyoruz
    for v in range(n_vehicles):
        # Rastgele iki nokta seçip bu aralıktaki genleri (düğümleri) çaprazlıyoruz
        start, end = sorted(random.sample(range(len(parent1[v])), 2))
        # Parent1'den gelen genler çocuk1'e eklenir, geri kalan kısımlar Parent2'den gelir
        child1.append(parent1[v][start:end] + [x for x in parent2[v] if x not in parent1[v][start:end]])
        # Aynı işlemi çocuk2 için de tersine yapıyoruz
        child2.append(parent2[v][start:end] + [x for x in parent1[v] if x not in parent2[v][start:end]])
    return child1, child2  # Oluşturulan iki yeni çocuk çözümünü döndür

# Mutasyon (Mutation): Çözüme rastgele değişiklikler yapıyoruz
# Mutasyon, çözümde küçük rastgele değişiklikler yaparak popülasyondaki çeşitliliği korur.
# Aksi takdirde algoritma, yerel minimuma sıkışabilir.
def mutate(solution):
    vehicle = random.randint(0, n_vehicles - 1)  # Rastgele bir araç seç
    if len(solution[vehicle]) > 1:
        # Araç rotasındaki iki düğümün yerini değiştir
        i, j = random.sample(range(len(solution[vehicle])), 2)
        solution[vehicle][i], solution[vehicle][j] = solution[vehicle][j], solution[vehicle][i]
    return solution  # Mutasyona uğramış çözümü döndür

# Genetik Algoritmanın ana fonksiyonu
# Bu fonksiyon tüm süreci içerir: popülasyonun başlatılması, seçilim, çaprazlama, mutasyon ve yeni nesil üretimi.
def genetic_algorithm():
    population = initialize_population()  # İlk popülasyonu rastgele oluşturuyoruz
    best_solution = min(population, key=fitness)  # Başlangıçta en iyi çözümü buluyoruz
    for _ in range(max_iterations):
        new_population = []  # Yeni nesil için boş bir popülasyon oluştur
        for _ in range(population_size // 2):
            # Seçilim işlemi: en iyi iki bireyi buluyoruz
            parent1 = selection(population)
            parent2 = selection(population)
            # Çaprazlama ile iki yeni çocuk oluşturuyoruz
            child1, child2 = crossover(parent1, parent2)
            # Mutasyon şansı %10, bu yüzden rastgele mutasyon yapıyoruz
            if random.random() < 0.1:
                child1 = mutate(child1)
            if random.random() < 0.1:
                child2 = mutate(child2)
            # Yeni nesil popülasyona çocukları ekliyoruz
            new_population.extend([child1, child2])
        # Mevcut popülasyon ile yeni popülasyonu birleştirip uygunluklarına göre sıralıyoruz
        population = sorted(population + new_population, key=fitness)[:population_size]
        best_solution = min(population, key=fitness)  # En iyi çözümü güncelliyoruz
    return best_solution, fitness(best_solution)  # En iyi çözüm ve onun mesafesini döndür

# Genetik algoritmayı çalıştır
best_solution, best_distance = genetic_algorithm()

print("En iyi çözüm:", best_solution)
print("En iyi mesafe:", best_distance)
