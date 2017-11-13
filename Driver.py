'''
Sam Congdon, Kendall Dilorenzo, Michel Hewitt
CSCI 447: MachineLearning
Project 3: Driver
November 13, 2017

This python module handles the testing of the for neural net training algorithms. Tests data sets using k-fold
cross validation, outputs the networks error throughout training and on the final test set on line graphs, one for
each fold. Output can be saved to a PDF.
'''

import NN
import GA
import ES
import DE
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
import os

def import_data(file_name):
    ''' Imports data points from supplied file, formatting them into data point pairs to be
        used by the algorithms. '''
    fin = open(file_name, 'r')
    input_line = fin.readline()
    data = []
    inputs = 0
    outputs = 0
    read_shape = False

    while input_line:
        input_line = input_line.strip().split(',')
        output_line = fin.readline().strip().split(',')
        if not read_shape:
            inputs = len(input_line)
            outputs = len(output_line)
            read_shape = True
        if input_line == [''] or output_line == ['']: break
        for i in range(len(input_line)): input_line[i] = float(input_line[i])
        for i in range(len(output_line)): output_line[i] = float(output_line[i])
        data.append((input_line, output_line))
        input_line = fin.readline()
    fin.close()
    return data, inputs, outputs, os.path.splitext(file_name)[0][9:] # last return is file name, assumes file is in datasets/ directory

def play_buzzer():
    ''' Plays a buzzer, used during testing for completion alerts. '''
    import pygame
    pygame.init()
    song = pygame.mixer.Sound('buzzer.wav')
    clock = pygame.time.Clock()
    song.play()
    clock.tick(1)
    pygame.quit()

def rosenbrock(x, y):
    return (1 - x) ** 2 + 100 * ((y - x ** 2)) ** 2

def create_data(num_points, min, max, dimensions = 2):
    ''' Creates data points from the rosenbrock function, input desired number of points, range of points, and dimension
        of the function. '''
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

# load data sets from files
data_iris = import_data('datasets/iris.txt')
#data_cmc = import_data('datasets/cmc.txt')
#data_abalone = import_data('datasets/abalone.txt')
#data_airfoil = import_data('datasets/airfoilnoise.txt')
#data_poker = import_data('datasets/pokerhand.txt')
#data_yeast = import_data('datasets/yeast.txt')
#dataset = create_data(10, -1, 1, 2)

# alter these parameters for general testing
dataset, inputs, outputs, data_name = data_iris
shuffle(dataset)
shape = [inputs,25,25,outputs]
k = 5   # cannot be larger than 6 without altering the figure's subplot layout

# create folds for cross validation
slices = []
length = int(len(dataset) / k)
for i in range(k-1):
    slices.append(dataset[i*length:(i+1)*length])
slices.append(dataset[(k-1)*length:])

plot_count = 1
fig = plt.figure(figsize=(14, 7))

# for each fold, rotates through testing sets
for i in range(k):
    test_data = slices.pop(0)
    train_data = [j for i in slices for j in i]
    slices.append(test_data)

    # train network, receive error throughout training and the trained network. Uncomment line to train a network
    hero, lifetime_error = NN.train(shape=shape, iterations=2000, data=train_data, learning_rate = 0.01, print_frequency=100)
    #hero, lifetime_error = GA.train(shape=shape, mu=15, generations=2000, number_possible_parents=4, number_parents=4, quick=False, data=train_data, print_frequency=100)
    #hero, lifetime_error = ES.train(shape=shape, mu=25, generations=2000, number_possible_parents=5, number_parents=5, quick=False, data=train_data, print_frequency=100)
    #hero, lifetime_error = DE.train(shape=shape, mu=20, generations=2000, data=train_data, print_frequency=100)

    test_error = hero.test(test_data)
    print("Testing error on fold {} = {}".format(i+1, test_error))

    # adds the fold's graph to the output figure
    fig.add_subplot(230+plot_count)
    plot_count += 1
    plt.title(lifetime_error[0] + ' on dataset {}'.format(data_name))
    plt.ylabel('Mean Squared Error')
    plt.xlabel('Generation')
    plt.grid(True)
    plt.plot(lifetime_error[1:])
    plt.plot(len(lifetime_error)-1, test_error, 'bs')

plt.tight_layout()
fig.savefig('figures/' + lifetime_error[0] + ' on dataset {}'.format(data_name) + '.pdf')
#plt.show()

#play_buzzer()

