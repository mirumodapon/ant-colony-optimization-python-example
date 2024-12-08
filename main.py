from random import randint, choices
from math import log10
from copy import deepcopy

# ''' ===== Figure ===== '''
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Configure constance value
CITY = 5
ITERATION = 10
ANT = 10

ALPHA = 1 # 2
BETA = 1 # 5
Q = 50 # 100
EVAPORATION_RATE = 0.5 # 0.3

# Define utils methods
calculate_distance = lambda a, b: ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

def printArr(arr, label = '', padding=7.2):
  size = len(arr)
  label_len = len(label)
  label_padding = max(label_len, int(log10(size))) + 3

  print(label + ' ' * (label_padding - label_len), end='|')
  for i in range(size):
    print(f'{i + 1: >{int(padding)}}', end='|')
  for i in range(size):
    print()
    print(f'{i + 1: >{label_padding}}', end='|')
    for j in arr[i]:
      print(f'{j: >{padding}f}', end='|')
  print()

def calculate_probabilities(current, unvisited, distance, pheromone):
  prob = []
  for next in unvisited:
    pher = pheromone[current][next] ** ALPHA
    heuristic = 1 / distance[current][next] ** BETA
    prob.append(pher * heuristic)

  total = sum(prob)
  prob = [p / total for p in prob]
  return prob

def update_pheromone(routes, distance, pheromone):
  pher = deepcopy(pheromone)

  for i in range(len(pher)):
    for j in range(len(pher[i])):
      if (i != j):
        pher[i][j] *= (1 - EVAPORATION_RATE)

  for route, cost in routes:
    contribution = Q / cost
    for i in range(CITY):
      pher[route[i]][route[i + 1]] += contribution
      pher[route[i + 1]][route[i]] += contribution

  return pher

# Generate random coordinate
cities = [(randint(0, 100), randint(0, 100)) for i in range(CITY)]

# ''' ===== Figure ===== '''
plt.figure(figsize=(8, 8))
x_coords = [coord[0] for coord in cities]
y_coords = [coord[1] for coord in cities]

plt.scatter(x_coords, y_coords, c = 'red', zorder = 5)

for i in range(CITY):
  plt.text(x_coords[i] + 0.5, y_coords[i], f'{i}', fontsize=12, color='blue')

plt.show()

# Calculate distance between cities
distance = [[calculate_distance(cities[i], cities[j]) for j in range(CITY)] for i in range(CITY)]
printArr(distance, 'D')

# Initialize pheromone
pheromone = [[1 if i != j else 0 for i in range(CITY)] for j in range(CITY)]
printArr(pheromone, 'P')

all_best = (None, float('Inf'))

# Start iteration
for i in range(ITERATION):
  print(f'Iteration {i + 1} Start')
  all_routes = []
  best_route = (None, float('Inf'))


  for ant in range(ANT):
    route = [randint(0, CITY - 1)]
    unvisited = set(range(CITY)) - set(route)

    while unvisited:
      current = route[-1]
      prob = calculate_probabilities(current, unvisited, distance, pheromone)
      [next] = choices(list(unvisited), prob)
      route.append(next)
      unvisited.remove(next)

    route.append(route[0])
    cost = sum(distance[route[i]][route[i + 1]] for i in range(CITY))
    all_routes.append((route, cost))

    print('\t', f'{ant: 3}) ',' -> '.join(str(c) for c in route), ', ', f'{cost: 3.2f}')

    if cost < best_route[1]:
      best_route = (route, cost)

  if best_route[1] < all_best[1]:
    all_best = best_route

  pheromone = update_pheromone(all_routes, distance, pheromone)
  print(f'\tBest Cost = {best_route[1]: 3.2f}')

print(f'Solve: {" -> ".join(str(c) for c in all_best[0])} {all_best[1]: 3.2f}')



# ''' ===== Figure route ===== '''
plt.figure(figsize=(8, 8))
x_coords = [coord[0] for coord in cities]
y_coords = [coord[1] for coord in cities]
plt.scatter(x_coords, y_coords, c = 'red', zorder = 5)


for i in range(CITY):
  plt.text(x_coords[i] + 0.5, y_coords[i], f'{i}', fontsize=12, color='blue')

x_coords = []
y_coords = []
for i in all_best[0][: -1]:
  x_coords.append(cities[i][0])
  y_coords.append(cities[i][1])

plt.plot(x_coords, y_coords, c='blue')
plt.show()

# ''' ===== Figure pheromone ===== '''
plt.figure(figsize=(8, 8))
x_coords = [coord[0] for coord in cities]
y_coords = [coord[1] for coord in cities]
plt.scatter(x_coords, y_coords, c = 'red', zorder = 5)


for i in range(CITY):
  plt.text(x_coords[i] + 0.5, y_coords[i], f'{i}', fontsize=12, color='blue')

total_pheromone = max(max(i) for i in pheromone) / 2
cmap = plt.get_cmap('Blues')
norm = mcolors.Normalize(vmin=0, vmax=total_pheromone)
for i in range(CITY):
  for j in range(CITY):
    if (i != j):
      plt.plot([x_coords[i], x_coords[j]], [y_coords[i], y_coords[j]], color=cmap(norm(pheromone[i][j])))

plt.show()
