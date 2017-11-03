import NN
import GA
import numpy as np
from operator import itemgetter
import math


def create_es_population(shape, mu):
    population = GA.create_population(shape, mu)
    for i in range(len(population)):
        individual = population[i][:-1]
        individual = np.concatenate((individual, np.random.normal(0, 1, len(individual)+1)))
        population[i] = individual
        #population[i][-1] = 0
    return population


def mutate(child, mutation_rate):
    ''' Mutates a supplied child. Each value in the vector has an equal chance of being mutated. '''
    offset = int((len(child)-1)/2)
    for i in range(offset):
        if np.random.random() < mutation_rate:
            # first mutate the associated sigma value
            child[i + offset] = abs(child[i + offset] * math.exp(np.random.normal(0, 1) / math.sqrt((len(child) - 1) / 2)))
            # then mutate the actual value of the child
            child[i] = child[i] + np.random.normal(0, child[i + offset])


def train(shape, mu, generations, number_possible_parents, number_parents, data, target_fitness = 0.00001, mutation_rate = 0.5, crossover_rate = 0.5):
    ''' Trains a network using the mu + lambda evolution strategy. Returns the best network created after either the max
        number of generations have been run, or the target_fitness has been achieved. '''

    # create and evaluate the original seed population
    population = create_es_population(shape, mu)
    GA.evaluate(shape, population, data)

    for i in range(generations+1):

        parents = GA.tournament(population, number_possible_parents, number_parents)
        #print("Tournament selected {} parents".format(number_parents))
        children = GA.reproduce(parents, crossover_rate)
        #print("Parents reproduced {} children".format(len(children)))

        for child in children:
            #print("Child before mutation = {}".format(child))
            mutate(child, mutation_rate)
            #print("Child after mutation = {}".format(child))

        GA.evaluate(shape, children, data)

        # keeps the best individuals from the combined children and population arrays to act as the
        # next generations population
        if len(children) != 0: population = np.concatenate((population, children))
        population = sorted(population, key=itemgetter(-1))[:mu]

        if population[0][-1] < target_fitness: break

        if i%250 == 0:
            print("Generation {}'s fitnesses are: ".format(i), end='')
            for i in population:
                print("{}, ".format(i[-1]), end = '')
            print()

    return population[0]

# -------------------Functionality below here is used for testing and configuration-------------------------------------

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * ((y - x ** 2)) ** 2

def create_data(num_points, min, max, dimensions = 2):
    input_sets = [np.random.uniform(min, max, dimensions) for _ in range(num_points)]
    data_set = []
    for input in input_sets:
        output = sum([rosenbrock(x, y) for x, y in zip(input[:-1], input[1:])])
        data_set.append([input, [output]])
    return data_set

data1 = [([2.7810836, 2.550537003], [0]),
           ([1.465489372, 2.362125076], [0]),
           ([3.396561688, 4.400293529], [0]),
           ([1.38807019, 1.850220317], [0]),
           ([3.06407232, 3.005305973], [0]),
           ([7.627531214, 2.759262235], [1]),
           ([5.332441248, 2.088626775], [1]),
           ([6.922596716, 1.77106367], [1]),
           ([8.675418651, -0.242068655], [1]),
           ([7.673756466, 3.508563011], [1])]

data = [([0,0], [1]), ([1,1], [1]), ([0,1], [0]), ([1,0], [0])]

data2 = [([0,0], [1,0]), ([1,1], [1,0]), ([0,1], [0,1]), ([1,0], [0,1])]

#data i requires 4 input and 3 output nodes
datai = NN.import_data('iris.txt')

#dataset = create_data(10, -1, 1, 2)

# alter these parameters for simpler testing
dataset = datai
shape = [4,5,3]

hero = train(shape, 20, 1000, 5, 5, dataset)

nn = NN.Network(shape)
counter = 0
for layer in nn.weights:
    for node in layer:
        for i in range(len(node)):
            node[i] = hero[counter]
            counter += 1
nn.output_test(dataset)


