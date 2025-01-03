
import math
import struct
from random import random
import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
import time

#parameter adjustment
POPULATION_SIZE = 200
BIT = 16
MUTATION_RATE = 0.25
CROSS_OVERRATE = 0.2
NO_GENERATION = 10
GOAL = 0.08
SAME_FOR_HOWMANY_GEN = 300
weight_bias = 100
TYPE_TERMINATION = "S" #N for by generation number, G for by fitness value, # S for times of repeating solution
TYPE_SELECTION = "t"  # r for roulette wheel; t for tournament
TYPE_CROSSOVER = 1  # 1 for single; 2 for multiple; 3 for uniform
MIN = 0
MAX = 20


def population_generator(size):
    return np.random.randint(0, 2, size)

class Individual:
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.square = self.normalize(self.chromosome[0])
        self.triangle = self.normalize(self.chromosome[1])
        self.circle = self.normalize(self.chromosome[2])
        self.fitness = self.fitness_function()

    def __lt__(self, other):
        return self.fitness < other.fitness

    def bit_to_dec(self, value):
        return sum(val * (2 ** idx) for idx, val in enumerate(reversed(value)))


    def normalize(self, value):
        return (MIN + (self.bit_to_dec(value)* (MAX-MIN))/(2**BIT - 1))


    def getArea(self):
        return (self.square / 4) ** 2 + ((math.sqrt(3) / 4) * ((self.triangle / 3) ** 2)) + (math.pi * (self.circle/ (2*math.pi))**2)

    def fitness_function(self):
        # contraint
        penalty_function = (abs(self.square - self.triangle) +
                            abs(self.square + self.triangle + self.circle - 20)) * weight_bias

        objective_function = self.getArea() + penalty_function
        return 1 / (1 + (abs(objective_function)))



#cross over function
def single_point_crossover(first, other, point):
    child_A = np.append(first[:point], other[point:])
    child_B = np.append(other[:point], first[point:])
    return child_A, child_B

def multi_point_crossover(first, other, point):
    child_A = np.copy(first)
    child_B = np.copy(other)
    for i in point:
        child_A, child_B = single_point_crossover(child_A, child_B, i)
    return child_A, child_B

def uniform_crossover(first, other, probability):
    child_A = np.copy(first)
    child_B = np.copy(other)

    for i in range(len(probability)):
        if probability[i] < 0.5:
            temp = child_A[i]
            child_A[i] = child_B[i]
            child_B[i] = temp
    return child_A, child_B

#selection function
def roulette_wheel_selection(population):
    total = sum([c.fitness for c in population])
    selection_probs = [c.fitness / total for c in population]
    return population[np.random.choice(len(population), p=selection_probs)]

def tournament_selection(population):
    temp = []
    temp.extend(population)
    winner = []

    for i in range(int(len(population) / 2)):
        compare = []
        rng = default_rng()
        two_random = rng.choice(len(temp), size=2, replace=False)
        two_random = np.sort(two_random)
        compare.append(temp[two_random[0]])
        compare.append(temp[two_random[1]])
        temp.pop(two_random[1])
        temp.pop(two_random[0])
        compare.sort(reverse=True)
        winner.append(compare[0])
        
    if (len(temp) == 1):
        winner.append(temp[0])
    return winner

#mutation function
def mutation(chromo, num=1, m_rate=0.5):
    for i in range(num):
        index = random.randrange(len(chromo))
        chromo[index] = chromo[index] if random.random() > m_rate else abs(chromo[index] - 1)
    return chromo

#To plot graph
def plot_graph(fitness):
    plt.title("Total generation " + str(NO_GENERATION), fontsize=16)
    ypoints = fitness
    xpoints = np.arange(1, len(fitness) + 1)
    plt.plot(xpoints, ypoints)
    plt.xlabel("Number of Generation")
    plt.xlabel("Fitness Score")
    plt.show()

def subplot_graph(fitness1, fitness2):
    plt.title("Total generation ", fontsize=16)
    plt.xlabel("Number of Generation")
    plt.ylabel("Fitness Score")
    plt.legend()

    xpoints = np.arange(1, len(fitness1) + 1)
    plt.plot(xpoints, fitness1, label="Top")
    plt.plot(xpoints, fitness2, label="Average")
    plt.legend()
    plt.show()

#main function
def main():
    #get intial run time
    st = time.time()
    #generate population of 3 set list of chromosome with BIT length of gene
    population = []
    for i in range(POPULATION_SIZE):
        population.append(Individual(population_generator([3, BIT])))

    counter = 1
    each_genTop = np.array([])
    each_genAverage = np.array([])
    last_gen = 0
    counter_gen = 0

    #flag to end the loop
    flag = True
    while flag:
        #selection
        if (TYPE_SELECTION == "r"):  # roulette wheel
            potential_parent = np.array([])
            for i in range(0, POPULATION_SIZE):
                potential_parent = np.append(potential_parent, roulette_wheel_selection(population))
        else:  # tournament
            potential_parent = np.array([tournament_selection(population)])
            if (len(potential_parent) == 1):
                potential_parent = population

        # determine number of chromosome for crossover
        cross_breed = np.array([])
        for indi in potential_parent:
            if (random.uniform(0, 1) < CROSS_OVERRATE):
                cross_breed = np.append(cross_breed, indi)
        offspring = np.array([])
        parent = list(itertools.combinations(cross_breed, 2))

        #cross over
        if (TYPE_CROSSOVER == 1):  # single point
            for x in parent:
                point = random.randint(1, (len(x[0].chromosome[0]) ) - 1)
                s1, s2 = single_point_crossover(x[0].chromosome[0], x[1].chromosome[0], point)
                t1, t2 = single_point_crossover(x[0].chromosome[1], x[1].chromosome[1], point)
                c1, c2 = single_point_crossover(x[0].chromosome[2], x[1].chromosome[2], point)

                child_A = (np.array([mutation(s1, random.randint(0, len(s1) - 1), MUTATION_RATE),
                                mutation(t1, random.randint(0, len(t1) - 1), MUTATION_RATE),
                                mutation(c1, random.randint(0, len(c1) - 1), MUTATION_RATE)]))

                child_B = (np.array([mutation(s2, random.randint(0, len(s2) - 1), MUTATION_RATE),
                                     mutation(t2, random.randint(0, len(t2) - 1), MUTATION_RATE),
                                     mutation(c2, random.randint(0, len(c2) - 1), MUTATION_RATE)]))
                offspring = np.append(offspring, Individual(child_A))
                offspring = np.append(offspring, Individual(child_B))
        elif (TYPE_CROSSOVER == 2):  # multiple point
            for x in parent:
                point = np.array([4, 5])
                s1, s2 = multi_point_crossover(x[0].chromosome[0], x[1].chromosome[0], point)
                t1, t2 = multi_point_crossover(x[0].chromosome[1], x[1].chromosome[1], point)
                c1, c2 = multi_point_crossover(x[0].chromosome[2], x[1].chromosome[2], point)

                child_A = (np.array([mutation(s1, random.randint(0, len(s1) - 1), MUTATION_RATE),
                                     mutation(t1, random.randint(0, len(t1) - 1), MUTATION_RATE),
                                     mutation(c1, random.randint(0, len(c1) - 1), MUTATION_RATE)]))

                child_B = (np.array([mutation(s2, random.randint(0, len(s2) - 1), MUTATION_RATE),
                                     mutation(t2, random.randint(0, len(t2) - 1), MUTATION_RATE),
                                     mutation(c2, random.randint(0, len(c2) - 1), MUTATION_RATE)]))
                offspring = np.append(offspring, Individual(child_A))
                offspring = np.append(offspring, Individual(child_B))
        else:
            for x in parent:  # uniform point
                point=(np.random.rand((len(x[0].chromosome[0]))))
                s1, s2 = uniform_crossover(x[0].chromosome[0], x[1].chromosome[0], point)
                t1, t2 = uniform_crossover(x[0].chromosome[1], x[1].chromosome[1], point)
                c1, c2 = uniform_crossover(x[0].chromosome[2], x[1].chromosome[2], point)

                child_A = (np.array([mutation(s1, random.randint(0, len(s1) - 1), MUTATION_RATE),
                                     mutation(t1, random.randint(0, len(t1) - 1), MUTATION_RATE),
                                     mutation(c1, random.randint(0, len(c1) - 1), MUTATION_RATE)]))

                child_B = (np.array([mutation(s2, random.randint(0, len(s2) - 1), MUTATION_RATE),
                                     mutation(t2, random.randint(0, len(t2) - 1), MUTATION_RATE),
                                     mutation(c2, random.randint(0, len(c2) - 1), MUTATION_RATE)]))
                offspring = np.append(offspring, Individual(child_A))
                offspring = np.append(offspring, Individual(child_B))

        population.extend(offspring)
        # remove duplicate
        remove_dup = set(population)
        population = list(remove_dup)
        #sort by descending order
        population.sort(reverse=True)
        temp = population[0:POPULATION_SIZE]  # pick the top best
        population = temp  # pick top best

        each_genTop = np.append(each_genTop, population[0])
        vectorized_x = np.vectorize(lambda obj: obj.fitness)
        each_genAverage = np.append(each_genAverage, np.average(vectorized_x(population)))
        print("-----------------------------------------------")
        print('Generation: {}\nTop fitness: {:.4f}, Minimum area: {:.4f}, '
              'Square perimeter: {:.4f}, Triangle perimeter: {:.4f}, Circle Circumferences: {:.4f}'
                .format(counter, population[0].fitness_function(),population[0].getArea(),population[0].square,population[0].triangle,population[0].circle))

        constraint1 = abs(population[0].square - population[0].triangle)
        constraint2 = abs(population[0].square + population[0].triangle + population[0].circle - 20)
        totalPeri = population[0].square + population[0].triangle + population[0].circle
        if(constraint1 == 0 and constraint2 == 0):
            print("All constraint meant")
        else:
            print("Total perimeter at: ", end = "")
            print(totalPeri)
            print("Square and Triangle perimeter differences: ", end = "")
            print(constraint1)

        if (last_gen == population[0].fitness):
            counter_gen += 1
        else:
            last_gen = population[0].fitness
            counter_gen = 0
            
        #termination method
        if(TYPE_TERMINATION == "N"): #by number of generation
            if (counter >= NO_GENERATION):
                flag = False
        elif(TYPE_TERMINATION == "G"): #by fitness goal
            if (population[0].fitness >= GOAL):
                flag = False
        elif(TYPE_TERMINATION == "S"):#by numbers of repeating result
            if (counter_gen >= SAME_FOR_HOWMANY_GEN - 1):
                flag = False

        counter += 1

    #end time
    et = time.time()
    elapsed_time = et - st
    #get execution time
    print('Execution time:', elapsed_time, 'seconds')
    #plot
    vectorized_x = np.vectorize(lambda obj: obj.fitness)
    subplot_graph(vectorized_x(each_genTop), each_genAverage)


if __name__ == '__main__':
    main()
