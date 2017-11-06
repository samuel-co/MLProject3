import ES
import GA
import NN
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

    while input_line:
        input_line = input_line.strip().split(',')
        output_line = fin.readline().strip().split(',')
        if input_line == [''] or output_line == ['']: break
        for i in range(len(input_line)): input_line[i] = float(input_line[i])
        for i in range(len(output_line)): output_line[i] = float(output_line[i])
        data.append((input_line, output_line))
        input_line = fin.readline()
    fin.close()
    return data, os.path.splitext(file_name)[0][9:] # second return is file name, assumes file is in datasets/ directory

def play_buzzer():
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

datai = import_data('datasets/iris.txt')
#dataset = create_data(10, -1, 1, 2)


dataset, data_name = datai
shuffle(dataset)
shape = [4,5,3]
k = 5   # cannot be larger than 6 without altering the figure's subplot layout

slices = []
length = int(len(dataset) / k)
for i in range(k-1):
    slices.append(dataset[i*length:(i+1)*length])
slices.append(dataset[4*length:])

plot_count = 1
fig = plt.figure(figsize=(14, 7))

for i in range(k):
    test_data = slices.pop(0)
    train_data = [j for i in slices for j in i]
    slices.append(test_data)

    hero, lifetime_error = NN.train(shape=shape, iterations=10, data=train_data, learning_rate = 0.01, print_frequency=250)
    #hero, lifetime_error = ES.train(shape=shape, mu=20, generations=10, number_possible_parents=5, number_parents=5, data=train_data, print_frequency=250)

    test_error = hero.test(test_data)
    print("Testing error on fold {} = {}".format(i+1, test_error))

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
plt.show()

#play_buzzer()


