#################################
Note by Rolf van Lieshout: below you find the original description of format, which is for an electric pickup and delivery problem. 
In our problem, we ignore this data. In addition, we only read in the first depot and ignore the others. 


#################################

The instances are formatted as follows:

###For each location the instance provides:
-StringId as a unique identifier
-Type indicates the function of the location, i.e,
---d: depot
--cp: customer pickup location
--cd: customer delivery location
-x, y are coordinates (distances are assumed to be euclidean) 
-demand specifies the quantity of freight capacity required (positive at pickup, negative at delivery)
-ReadyTime and DueDate are the beginning and the end of the time window (waiting is allowed)
-ServiceTime denotes the entire time spend at pickup delivery for loading unloading operations
-PartnerId is relevant for transportation requests and provides the StringId of the partner of each pickup and delivery location

###For the electric vehicles (all identical):
-VehicleCount: count of vehicle
-VehiclePerCapacity: load capacity of the trolley adding by each vehicle
-VehicleMaxTrolleyCount: Maximum number of trolleys to be fitted by each vehicle
-TrolleyImpactRate:  coefficient cost of installing each trolley
-EarlinessPenalty: vehicle's early arrival penalty point
-TardinessPenalty: vehicle's late arrival penalty point




