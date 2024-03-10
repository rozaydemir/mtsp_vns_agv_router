import matplotlib.pyplot as plt
import random
import numpy as np
import time


# Şehrin x, y koordinatlarını okuyun
def load(txt):
    f = open(txt)
    map = []
    flag = 0
    for line in f:
        line = line.strip()
        if line == "NODE_COORD_SECTION":
            flag = 1
            continue
        if line == "EOF":
            break
        if flag:
            a = line.split()
            map.append((float(a[1]), float(a[2])))
    return tuple(map)


# İki şehir arasındaki 2B Öklid mesafesini alın
# t değişkenini oluşturacak
def getDist():
    global map, size
    dist = np.zeros((size, size))
    for i in range(0, size):
        for j in range(0, size):
            dist[i][j] = ((map[i][0] - map[j][0]) ** 2 + (map[i][1] - map[j][1]) ** 2) ** 0.5
    return dist


txt = ".\\a280.txt"
map = load(txt)
size = len(map)
visited = {}
solutions = []
DIST = getDist()
count = 0


# Yolu temel alarak yolun toplam maliyetini alın
def getCost(path):
    cost = 0
    former = path[0]
    for city in path:
        cost += DIST[former][city]
        former = city
    cost += DIST[path[0]][path[-1]]
    return cost


# Pertürbasyon, rastgele sıralanmış dört aralığa bölünecek şekilde pertürbe edilmiş yeni bir rastgele çözüm üretir
def shaking(path):
    global size
    ini = visited[path]
    cnt = 0
    while True:
        pos1, pos2, pos3 = sorted(random.sample(range(0, size), 3))
        path_ = path[pos1:pos2] + path[:pos1] + path[pos3:] + path[pos2:pos3]
        if path_ not in visited:
            cost = getCost(path_)
            visited.update({path_: cost})
        else:
            cost = visited[path_]
        cnt += 1
        if ini >= cost:
            break
        elif cnt > 100:
            path_ = path
            cost = ini
            break
    return path_


# Yeni bir komşuluk elde etmek için bir aralığı tersine çevirin
def getNei_rev(path):
    global size
    min = visited[path]
    cnt = 0
    while True:
        i, j = sorted(random.sample(range(1, size - 1), 2))
        path_ = path[:i] + path[i:j + 1][::-1] + path[j + 1:]
        if path_ not in visited:
            cost = getCost(path_)
            visited.update({path_: cost})
        else:
            cost = visited[path_]
        cnt += 1
        if cost < min:
            min = cost
            break
        elif cnt > 1000:
            path_ = path
            break
    return path_, min


# Yeni bir mahalle almak için iki şehri değiştirin
def getNei_exc(path):
    global size
    min = visited[path]
    cnt = 0
    while True:
        i, j = sorted(random.sample(range(1, size - 1), 2))
        path_ = path[:i] + path[j:j + 1] + path[i + 1:j] + path[i:i + 1] + path[j + 1:]
        if path_ not in visited:
            cost = getCost(path_)
            visited.update({path_: cost})
        else:
            cost = visited[path_]
        cnt += 1
        if cost < min:
            min = cost
            break
        elif cnt > 1000:
            path_ = path
            break
    return path_, min


# Yeni bir mahalle elde etmek için dizinin başına eklemek üzere rastgele iki şehir seçin
def getNei_ins(path):
    global size
    min = visited[path]
    cnt = 0
    while True:
        i, j = sorted(random.sample(range(1, size - 1), 2))
        path_ = path[i:i + 1] + path[j:j + 1] + path[:i] + path[i + 1:j] + path[j + 1:]
        if path_ not in visited:
            cost = getCost(path_)
            visited.update({path_: cost})
        else:
            cost = visited[path_]
        cnt += 1
        if cost < min:
            min = cost
            break
        elif cnt > 1000:
            path_ = path
            break
    return path_, min


# VND yöntemini kullanarak Yerel Arama'da arama yapma
def VND(path):
    l = 0
    min = visited[path]
    while l < 3:
        if l == 0:
            path_, cost = getNei_rev(path)
        elif l == 1:
            path_, cost = getNei_exc(path)
        elif l == 2:
            path_, cost = getNei_ins(path)
        if cost < min:
            path = path_
            min = cost
            l = 0
        else:
            l += 1
    return path, min


# Değişken komşuluk yerel aramaları gerçekleştirin
def VNS(path, kmax):
    k = 0
    temp = path
    min = solutions[0]
    global count
    while k < kmax:
        # Pertürbasyondan sonra değişken komşuluk işlemi
        path_nei, cost = VND(shaking(temp))
        print(cost)
        solutions.append(cost)
        count += 1
        if cost < min:
            temp = path_nei  # İterasyondan sonra optimum çözümü kaydedin
            min = cost
            k = 0
        else:
            k += 1
    return temp, min


def main():
    time_start = time.time()
    global solutions, visited, size, map
    kmax = 1000
    start = tuple([k for k in range(size)])
    visited.update({start: getCost(start)})
    solutions.append(visited[start])
    path_, cost = VNS(start, kmax)
    path = path_[:] + path_[:1]
    time_end = time.time()
    print()
    print('Algorithm VNS iterated', count, 'times!\n', sep=' ')
    print('It cost ', time_end - time_start, 's', sep='')  # Bu saniye cinsindendir.
    print('You got the best solution:', cost, sep='\n')
    print(path)
    best = int(input("The best solution should be: "))
    print("Hata şu şekildedir：", (cost - best) / best)
    x = np.array([map[i][0] for i in path])
    y = np.array([map[i][1] for i in path])
    i = np.arange(0, len(solutions))
    solutions = np.array(solutions)
    plt.subplot(121)
    plt.plot(x, y)
    plt.subplot(122)
    plt.plot(i, solutions)
    plt.show()


main()


