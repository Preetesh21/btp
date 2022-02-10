maximum_demand=50*10^6;
load_factor=0.75;
capacity_factor=0.5;
plant_use_factor=0.8;
average_load=load_factor*maximum_demand;
daily_energy_produced=average_load*24;
plant_capacity=average_load/capacity_factor;
reserve_capacity=plant_capacity-maximum_demand;
max_energy_produced=daily_energy_produced/plant_use_factor;
fprintf('Daily energy produced= %d.\n',daily_energy_produced);
fprintf('Daily working hours=24 \n');
fprintf('Reserve Capacity= %d.\n',reserve_capacity);
fprintf('Max energy produced= %d.\n',max_energy_produced);

