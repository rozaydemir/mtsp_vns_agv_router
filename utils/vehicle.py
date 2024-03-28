class Vehicle:
    def __init__(self, id, base_capacity, max_trolleys=3, trolley_capacity=50):
        self.id = id
        self.base_capacity = base_capacity
        self.max_trolleys = max_trolleys
        self.trolley_capacity = trolley_capacity
        self.trolleys_attached = 0
        self.routes = []
        self.load = 0

    @property
    def total_capacity(self):
        """Araç ve takılı trolleynin toplam kapasitesini hesaplar."""
        return self.base_capacity + self.trolleys_attached * self.trolley_capacity

    def add_customer(self, customer_demand):
        """Müşteriyi ekler; eğer gerekirse trolley takar."""
        if self.load + customer_demand <= self.total_capacity:
            self.routes.append(customer_demand)
            self.load += customer_demand
            return True
        elif self.trolleys_attached < self.max_trolleys and self.load + customer_demand <= (self.total_capacity + self.trolley_capacity):
            # Trolley ekle
            self.trolleys_attached += 1
            self.routes.append(customer_demand)
            self.load += customer_demand
            return True
        else:
            return False

# # Araçlar ve müşteriler
# vehicles = [Vehicle(id=1, base_capacity=100)]
# customers = [20, 30, 40, 50, 60, 70, 80]
#
# solution = create_solution(customers, vehicles)
# if solution:
#     for vehicle in solution:
#         print(f"Araç {vehicle.id}, yük: {vehicle.load}, rota: {vehicle.route}, takılı trolley: {vehicle.trolleys_attached}")
# else:
#     print("Verilen araçlarla tüm müşterilere hizmet verilemiyor.")
