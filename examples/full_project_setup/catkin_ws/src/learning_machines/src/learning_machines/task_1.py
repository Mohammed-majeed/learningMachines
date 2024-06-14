
from data_files import FIGRURES_DIR
import numpy as np
import math
from scipy import stats
import time
import random
import pickle
import os

from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)


def read_irs_sensors(rob, num_reads=7):
    joint_list = ["BackL", "BackR", "FrontL", "FrontR", "FrontC", "FrontRR", "BackC", "FrontLL"]
    no_obstacle_sens_values = [6.434948026696321, 6.4375698872759655, 52.26984940735039, 52.270314744820546, 5.845623601383301, 5.890924916422574, 57.76850943075616, 5.925058770384208]

    readings = {joint: [] for joint in joint_list}
    for _ in range(num_reads):
        temp = rob.read_irs()
        temp2 = []
        for i, val in enumerate(temp):
            if val != math.inf:          
                temp2.append(abs(temp[i]))
            else:
                temp2.append(abs(no_obstacle_sens_values[i]))
        irs_data = np.round(np.array(temp2) - np.array(no_obstacle_sens_values), decimals=1)
        for joint, value in zip(joint_list, irs_data):
            readings[joint].append(value)
    sensor_modes = {joint: stats.mode(values)[0][0] for joint, values in readings.items()}
    
    return sensor_modes




def move_forward(rob, speed, duration):
    """
    Move the robot forward.
    """
    rob.move_blocking(left_speed=speed, right_speed=speed, millis=duration)
    time.sleep(duration/1000)

def move_backward(rob, speed, duration):
    """
    Move the robot backward.
    """
    rob.move_blocking(left_speed=-speed, right_speed=-speed, millis=duration)
    time.sleep(duration/1000)

def turn_left(rob, speed, duration):
    """
    Turn the robot left.
    """
    rob.move_blocking(left_speed=-speed, right_speed=speed, millis=duration)
    time.sleep(duration/1000)

def turn_right(rob, speed, duration):
    """
    Turn the robot right.
    """
    rob.move_blocking(left_speed=speed, right_speed=-speed, millis=duration)
    time.sleep(duration/1000)






def fitness(individual, rob, start_position, start_orientation, target_position):
    rob.set_position(start_position, start_orientation)  
    collisions = 0
    distance_to_target = float('inf')

    threshold = 20

    print('individual',individual)
    for command in individual:
        
        if command == "move_forward":
            move_forward(rob, speed=50, duration=500)
        elif command == "turn_left":
            turn_left(rob, speed=50, duration=500)
        elif command == "turn_right":
            turn_right(rob, speed=50, duration=500)

        
        sensor_dict = read_irs_sensors(rob)
        if (sensor_dict["FrontC"] > threshold or
            sensor_dict["FrontR"] > threshold or
            sensor_dict["FrontL"] > threshold):

            collisions += 1

        
        current_position = rob.get_position()
        print('current_position',current_position)
        distance_to_target = ((current_position.x - target_position.x) ** 2 +
                              (current_position.y - target_position.y) ** 2) ** 0.5
        
    print(collisions)

    print('distance_to_target',distance_to_target)
        
    fit= -distance_to_target - (collisions * 10)  
    print(fit)

    return fit


def initialize_population(size):
    commands = ["move_forward", "turn_left", "turn_right"]
    return [[random.choice(commands) for _ in range(20)] for _ in range(size)]


def selection(population, fitnesses, num_parents):
    
    combined = list(zip(population, fitnesses))
    
    combined.sort(key=lambda x: x[1], reverse=True)
    
    selected_parents = [individual for individual, fitness in combined[:num_parents]]
    return selected_parents


def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(individual, mutation_rate=0.1):
    commands = ["move_forward", "turn_left", "turn_right"]
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = random.choice(commands)
    return individual



def save_checkpoint(population, fitnesses, best_individual, generation, filename='checkpoint.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump({
            'population': population,
            'fitnesses': fitnesses,
            'best_individual': best_individual,
            'generation': generation
        }, f)


def load_checkpoint(filename='checkpoint.pkl'):
    with open(filename, 'rb') as f:
        checkpoint = pickle.load(f)
    return checkpoint


def evolutionary_algorithm(rob, start_position, start_orientation, target_position,
                            generations=100, population_size=20,
                              checkpoint_file='checkpoint.pkl', continue_from_checkpoint=False):
    
    if continue_from_checkpoint:
        
        checkpoint = load_checkpoint(checkpoint_file)
        population = checkpoint['population']
        generation_start = checkpoint['generation'] + 1
        best_individual = checkpoint['best_individual']
    else:
        population = initialize_population(population_size)
        generation_start = 0
        best_individual = None

    for generation in range(generation_start, generations):
        fitnesses = [fitness(individual, rob, start_position, start_orientation, target_position) for individual in population]

        
        num_parents = max(1, population_size // 10)  
        parents = selection(population, fitnesses, num_parents)

        
        new_population = []
        while len(new_population) < (population_size - num_parents):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        
        population = new_population[:population_size - num_parents] + parents

        
        best_individual_index = fitnesses.index(max(fitnesses))
        best_individual = population[best_individual_index]
        save_checkpoint(population, fitnesses, best_individual, generation, checkpoint_file)

        
        best_fitness = max(fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")

    return best_individual













