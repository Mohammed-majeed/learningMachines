import numpy as np
import math
import time
import random
import json
import os
from scipy import stats

from data_files import RESULT_DIR

from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
)

class robot_controller:
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def set(self, controller, n_inputs):
        if self.n_hidden[0] > 0:
            self.bias1 = controller[:self.n_hidden[0]].reshape(1, self.n_hidden[0])
            weights1_slice = n_inputs * self.n_hidden[0] + self.n_hidden[0]
            self.weights1 = controller[self.n_hidden[0]:weights1_slice].reshape((n_inputs, self.n_hidden[0]))
            self.bias2 = controller[weights1_slice:weights1_slice + 3].reshape(1, 3)
            self.weights2 = controller[weights1_slice + 3:].reshape((self.n_hidden[0], 3))

    def control(self, inputs, controller):
        inputs = np.array(inputs)  # Convert inputs to a NumPy array
        input_min = min(inputs)
        input_max = max(inputs)
        if input_max - input_min != 0:
            inputs = (inputs - input_min) / float(input_max - input_min)
        else:
            inputs = inputs * 0

        if self.n_hidden[0] > 0:
            output1 = sigmoid_activation(inputs.dot(self.weights1) + self.bias1)
            output = sigmoid_activation(output1.dot(self.weights2) + self.bias2)[0]
        else:
            bias = controller[:3].reshape(1, 3)
            weights = controller[3:].reshape((len(inputs), 3))
            output = sigmoid_activation(inputs.dot(weights) + bias)[0]

        if output[0] > 0.5:
            return 'move_forward'
        elif output[1] > 0.5:
            return 'turn_left'
        elif output[2] > 0.5:
            return 'turn_right'
        else:
            return 'move_forward'


def sigmoid_activation(x):
    return 1. / (1. + np.exp(-x))

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
        # irs_data = np.round(np.array(temp2)), decimals=1)
        for joint, value in zip(joint_list, irs_data):
            readings[joint].append(value)
    sensor_modes = {joint: stats.mode(values)[0][0] for joint, values in readings.items()}

    front_sensors = {
        "FrontC": sensor_modes["FrontC"],
        "FrontR": sensor_modes["FrontR"],
        "FrontL": sensor_modes["FrontL"],
        "FrontRR": sensor_modes["FrontRR"],
        "FrontLL": sensor_modes["FrontLL"]
    }
    print(front_sensors)
    return front_sensors

# def read_irs_sensors(rob, num_reads=7):
#     joint_list = ["BackL", "BackR", "FrontL", "FrontR", "FrontC", "FrontRR", "BackC", "FrontLL"]
#     no_obstacle_sens_values = [6.434948026696321, 6.4375698872759655, 52.26984940735039, 52.270314744820546, 5.845623601383301, 5.890924916422574, 57.76850943075616, 5.925058770384208]

#     readings = {joint: [] for joint in joint_list}
#     for _ in range(num_reads):
#         temp = rob.read_irs()
#         temp2 = []
#         for i, val in enumerate(temp):
#             if val != math.inf:
#                 temp2.append(abs(temp[i]))
#             else:
#                 temp2.append(abs(no_obstacle_sens_values[i]))
#         irs_data = np.round(np.array(temp2), decimals=1)
#         for joint, value in zip(joint_list, irs_data):
#             readings[joint].append(value)
    
#     # sensor_modes = {joint: stats.mode(values)[0][0] for joint, values in readings.items()}
#     sensor_modes = {joint: round(stats.mode(values)[0][0], 1) for joint, values in readings.items()}

#     max_value = max(sensor_modes.values())

#     normalized_sensors = {joint: value / max_value for joint, value in sensor_modes.items()}

#     front_sensors = {
#         "FrontC": normalized_sensors["FrontC"],
#         "FrontR": normalized_sensors["FrontR"],
#         "FrontL": normalized_sensors["FrontL"],
#         "FrontRR": normalized_sensors["FrontRR"],
#         "FrontLL": normalized_sensors["FrontLL"]
#     }
#     print(front_sensors)
    
#     return front_sensors

def move_forward(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=speed, millis=duration)
    # time.sleep(duration/1000)

def turn_left(rob, speed, duration):
    rob.move_blocking(left_speed=-speed, right_speed=speed, millis=duration)
    # time.sleep(duration/1000)

def turn_right(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=-speed, millis=duration)
    # time.sleep(duration/1000)

def fitness(individual, rob, start_position, start_orientation, target_position, controller, steps):
    rob.set_position(start_position, start_orientation)  # Reset robot's position at the start of each evaluation
    controller.set(individual, n_inputs=5)  # Initialize the controller with the weights
    collisions = 0
    distance_to_target = float('inf')

    threshold = 250 # threshold for collisions

    for _ in range(steps):  # 20 steps for evaluation
        sensor_dict = read_irs_sensors(rob)
        # print('sensor_dict', sensor_dict)
        sensor_inputs = list(sensor_dict.values())
        action = controller.control(sensor_inputs, individual)
        print('action', action)

        if action == "move_forward":
            move_forward(rob, speed=50, duration=500)
        elif action == "turn_left":
            turn_left(rob, speed=50, duration=500)
        elif action == "turn_right":
            turn_right(rob, speed=50, duration=500)

        sensor_dict = read_irs_sensors(rob)
        if (sensor_dict["FrontC"] > threshold or
            sensor_dict["FrontR"] > threshold or
            sensor_dict["FrontL"] > threshold):
            collisions += 1

        current_position = rob.get_position()
        distance_to_target = ((current_position.x - target_position.x) ** 2 +
                              (current_position.y - target_position.y) ** 2) ** 0.5
        
    fit = -distance_to_target - (collisions * 10)  # Negative because we want to minimize this value
    print("fit",fit)
    return fit

def initialize_population(size, n_weights):
    return [np.random.uniform(-10, 10, n_weights) for _ in range(size)]

# def selection(population, fitnesses, num_parents):
#     combined = list(zip(population, fitnesses))
#     combined.sort(key=lambda x: x[1], reverse=True)
#     selected_parents = [individual for individual, fitness in combined[:num_parents]]
#     return selected_parents
def tournament_selection(population, fitnesses, num_parents, tournament_size=3):
    selected_parents = []
    for _ in range(num_parents):
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        tournament.sort(key=lambda x: x[1], reverse=True)
        selected_parents.append(tournament[0][0])
    return selected_parents

def crossover(parent1, parent2):
    crossover_point = random.randint(0, len(parent1))
    child1 = np.concatenate((parent1[:crossover_point], parent2[crossover_point:]))
    child2 = np.concatenate((parent2[:crossover_point], parent1[crossover_point:]))
    return child1, child2

def mutate(individual, mutation_rate=0.5):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += np.random.normal()
    return individual

def save_checkpoint_jsonl(population, fitnesses, best_individual, generation, filename):
    try:
        best_fitness = max(fitnesses)  # Calculate the best fitness
        checkpoint = {
            'generation': generation,
            'fitnesses': fitnesses,
            'best_fitness': best_fitness,  # Save the best fitness
            'best_individual': best_individual.tolist(),
            'population': [ind.tolist() for ind in population]
        }
        with open(filename, 'a') as f:  # Open in append mode
            f.write(json.dumps(checkpoint) + '\n')
        print(f"Checkpoint saved successfully at generation {generation}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")



def load_checkpoint_jsonl(filename):
    try:
        with open(filename, 'r') as f:
            last_line = None
            for last_line in f:  # Read the last line
                pass
            if last_line is None:
                raise ValueError("Checkpoint file is empty")
            checkpoint = json.loads(last_line)
            checkpoint['population'] = [np.array(ind) for ind in checkpoint['population']]
            checkpoint['best_individual'] = np.array(checkpoint['best_individual'])
            print(f"Checkpoint loaded successfully for generation {checkpoint['generation']}")
            return checkpoint
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return None
    

checkpoint_path = str(RESULT_DIR/"checkpoint.jsonl")
n_hidden = 2
n_inputs = 5  # Number of sensors
n_outputs = 3  # Number of actions

# Usage of the new selection function in the evolutionary_algorithm function
def evolutionary_algorithm(rob, start_position, start_orientation, target_position,
                           generations=100, population_size=5,
                           checkpoint_file=checkpoint_path, continue_from_checkpoint=True,
                           steps=20):

    n_weights = n_inputs * n_hidden + n_hidden * n_outputs + n_hidden + n_outputs

    controller = robot_controller(n_hidden)
    
    if continue_from_checkpoint:
        checkpoint = load_checkpoint_jsonl(checkpoint_file)
        if checkpoint:
            population = checkpoint['population']
            generation_start = checkpoint['generation'] + 1
            best_individual = checkpoint['best_individual']
        else:
            population = initialize_population(population_size, n_weights)
            generation_start = 0
            best_individual = None
    else:
        population = initialize_population(population_size, n_weights)
        generation_start = 0
        best_individual = None

    for generation in range(generation_start, generations):
        fitnesses = [fitness(individual, rob, start_position, start_orientation, target_position, controller, steps) for individual in population]
        
        num_parents = max(2, population_size // 10)  # Calculate 10% of population size, ensuring at least 2 parents
        parents = tournament_selection(population, fitnesses, num_parents, tournament_size=3)

        new_population = []
        while len(new_population) < (population_size - num_parents):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population[:population_size - num_parents] + parents

        best_individual_index = fitnesses.index(max(fitnesses))
        best_individual = population[best_individual_index]
        save_checkpoint_jsonl(population, fitnesses, best_individual, generation, checkpoint_file)

        best_fitness = max(fitnesses)
        print(f"Generation {generation}: Best Fitness = {best_fitness}")





###################
# test best individual 


def load_best_individual():
    try:
        with open(checkpoint_path, 'r') as f:
            last_line = None
            for last_line in f:  # Read the last line
                pass
            if last_line is None:
                raise ValueError("Checkpoint file is empty")
            checkpoint = json.loads(last_line)
            best_individual = np.array(checkpoint['best_individual'])
            print(f"Best individual loaded successfully from generation {checkpoint['generation']}")
            return best_individual
    except Exception as e:
        print(f"Error loading best individual: {e}")
        return None

def test_best_individual(rob, start_position, start_orientation, target_position, steps):
    controller = robot_controller(n_hidden)
    best_individual = load_best_individual()
    controller.set(best_individual, n_inputs)  # Initialize the controller with the best individual weights

    fit = fitness(best_individual, rob, start_position, start_orientation, target_position, controller, steps)

    # rob.set_position(start_position, start_orientation)  # Reset robot's position at the start of the test
    # distance_to_target = float('inf')
    # threshold = 75 # threshold for collisions
    # collisions = 0

    # for _ in range(steps):
    #     sensor_dict = read_irs_sensors(rob)
    #     sensor_inputs = list(sensor_dict.values())
    #     action = controller.control(sensor_inputs, best_individual)

    #     if action == "move_forward":
    #         move_forward(rob, speed=50, duration=500)
    #     elif action == "turn_left":
    #         turn_left(rob, speed=50, duration=500)
    #     elif action == "turn_right":
    #         turn_right(rob, speed=50, duration=500)

    #     sensor_dict = read_irs_sensors(rob)
    #     if (sensor_dict["FrontC"] > threshold or
    #         sensor_dict["FrontR"] > threshold or
    #         sensor_dict["FrontL"] > threshold):
    #         collisions += 1

    #     current_position = rob.get_position()
    #     distance_to_target = ((current_position.x - target_position.x) ** 2 +
    #                           (current_position.y - target_position.y) ** 2) ** 0.5

    # fit = -distance_to_target - (collisions * 10)  # Negative because we want to minimize this value
    print("Test fit:", fit)
    # return fit