import random

# Verilen veriler
vehicleCount = 2
capacityPerVehicle = [100, 100]  # Her bir aracın kapasitesi
depotLocation = (0, 0)  # Depo konumu
pickUpLocation = [(5, 10), (10, 15), (10, 15), (25, 20), (45, 10)]  # Alım noktaları
deliveryLocation = [(15, 10), (35, 30), (15, 15), (55, 25)]  # Teslimat noktaları
demandPickup = [5, 10, 22, 35, 25]  # Alım noktalarındaki yük miktarı
demandDelivery = [10, 22, 5, 35, 20]  # Teslimat noktalarındaki yük miktarı
serviceTime = [0, 10, 10, 10, 15, 5, 5, 5, 12, 22, 25]  # Servis süreleri

# İlk çözümü oluşturma
vehicles = []

# Her araç için bir rota oluştur
for vehicle_id in range(vehicleCount):
    route = [depotLocation]  # Her rotanın başlangıcı depo konumu
    load = 0

    # Bir alım noktası seç (her alım noktası benzersiz olmalı)
    while True:
        pick_index = random.randint(0, len(pickUpLocation) - 1)
        pick = pickUpLocation[pick_index]

        # Eğer kapasite aşılıyorsa, başka bir nokta deneyin
        if load + demandPickup[pick_index] > capacityPerVehicle[vehicle_id]:
            continue

        # Alım noktasını ve yükü rota ve yüke ekle
        route.append(pick)
        load += demandPickup[pick_index]
        break  # Bir alım noktası başarıyla eklendi

    # Bir teslimat noktası seç (her teslimat noktası benzersiz olmalı)
    while True:
        del_index = random.randint(0, len(deliveryLocation) - 1)
        delivery = deliveryLocation[del_index]

        # Eğer kapasite aşılıyorsa, başka bir nokta deneyin
        if load - demandDelivery[del_index] < 0:
            continue

        # Teslimat noktasını ve yükü rota ve yüke ekle
        route.append(delivery)
        load -= demandDelivery[del_index]
        break  # Bir teslimat noktası başarıyla eklendi

    # Rota sonunu depo ile bitir
    route.append(depotLocation)

    # Araç ve rotasını sonuç listesine ekle
    vehicles.append((vehicle_id, route))

# Sonuçları yazdır
for vehicle in vehicles:
    print(f"Araç {vehicle[0]}: Rota: {vehicle[1]}")
