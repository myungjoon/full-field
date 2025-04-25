import itertools


positions = ["on", "off"]
total_powers = [1.5e6, 1.6e6]

# combination of positions and total powers
combinations = list(itertools.product(positions, total_powers))
# loop for each combination
for position, total_power in combinations:
    print(f"Position: {position}, Total Power: {total_power}")