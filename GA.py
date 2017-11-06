import numpy as np
import random
from operator import itemgetter
import NN


def create_population(shape, mu):
    ''' Creates the initial population vectors of size mu, based on the network shape.
        The last element in each individual vector stores the network's fitness. '''
    population = []
    size = 0
    for i in range(len(shape)-1):
        size += shape[i] * shape[i+1]
    for i in range(mu):
        population.append(np.random.uniform(0, 0.1, size+1))
        #population[i][-1] = 0
    return population

def evaluate(shape, population, data):
    ''' Tests the supplied data on each individual in the population, storing the overall
            fitness of the netowork as the last element of each individual's array. '''
    for individual in population:
        nn = build_network(shape, individual)
        individual[-1] = nn.test(data)

def build_network(shape, individual):
    ''' Builds a neural net of the input shape using the individual's array of weights. '''
    nn = NN.Network(shape)
    counter = 0
    for layer in nn.weights:
        for node in layer:
            for i in range(len(node)):
                node[i] = individual[counter]
                counter += 1
    return nn


def tournament(population, number_possible_parents, number_parents):
    ''' Returns specified number of parents from the population using tournament selection.
        The commented out return can be used to return parents from a purely elitist selection. '''

    #return sorted(population, key=itemgetter(-1))[:number_parents]

    parents = []
    for i in range(number_parents):
        parents.append(sorted(random.sample(population, number_possible_parents), key=itemgetter(-1))[0])
    return parents

def tournament_unique(population, number_possible_parents, number_parents):
    ''' Returns unique parents of parents from the population using tournament selection.
        Returns , i.e. no parents can be selected more than once.'''

    parents = []
    for i in range(number_parents):
        parents.append(sorted(random.sample(population, number_possible_parents), key=itemgetter(-1))[0])
    return np.unique(parents, axis=0)


def reproduce(parents, crossover_rate):
    ''' Creates children from each combination of supplied parents, preventing duplication
        (i.e. the same parent can't mate with itself). '''
    children = []
    for parent1 in parents:
        for parent2 in parents:
            if not (parent1 == parent2).all():
                child1, child2 = crossover(parent1, parent2, crossover_rate)
                children.append(child1)
                children.append(child2)
    return children

def crossover(parent1, parent2, crossover_rate):
    ''' Performs uniform crossover on the supplied parents. '''
    child1 = parent1.copy()
    child2 = parent2.copy()
    for i in range(len(parent1) - 1):
        if random.random() < crossover_rate:
            child1[i] = parent2[i]
            child2[i] = parent1[i]
    return child1, child2


# Kendall everything i put in here i think you can reuse, not entirely sure though. You'll for sure need
# at least a new mutate and train method, i think the ES ones may be similar
