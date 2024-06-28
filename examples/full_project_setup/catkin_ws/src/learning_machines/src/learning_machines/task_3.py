import numpy as np
import math
import time
import random
import json
import os
from scipy import stats
from .img_proc import get_dist

from data_files import RESULT_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
    Position,
)

class robot_controller:
    def __init__(self, hidden_layers):
        self.hidden_layers = hidden_layers

    def set(self, controller, n_inputs):
        offset = 0
        
        
        if self.hidden_layers[0] > 0:
            self.bias1 = controller[offset:offset + self.hidden_layers[0]].reshape(1, self.hidden_layers[0])
            offset += self.hidden_layers[0]
            self.weights1 = controller[offset:offset + n_inputs * self.hidden_layers[0]].reshape((n_inputs, self.hidden_layers[0]))
            offset += n_inputs * self.hidden_layers[0]
        
        
        if len(self.hidden_layers) > 1 and self.hidden_layers[1] > 0:
            self.bias2 = controller[offset:offset + self.hidden_layers[1]].reshape(1, self.hidden_layers[1])
            offset += self.hidden_layers[1]
            self.weights2 = controller[offset:offset + self.hidden_layers[0] * self.hidden_layers[1]].reshape((self.hidden_layers[0], self.hidden_layers[1]))
            offset += self.hidden_layers[0] * self.hidden_layers[1]
        
        
        self.bias_output = controller[offset:offset + 3].reshape(1, 3)
        offset += 3
        self.weights_output = controller[offset:].reshape((self.hidden_layers[-1], 3))

    def control(self, inputs, controller):
        inputs = np.array(inputs)
        
        if self.hidden_layers[0] > 0:
            output1 = relu_activation(inputs.dot(self.weights1) + self.bias1)
            if len(self.hidden_layers) > 1 and self.hidden_layers[1] > 0:
                output2 = relu_activation(output1.dot(self.weights2) + self.bias2)
                output = softmax_activation(output2.dot(self.weights_output) + self.bias_output)[0]
            else:
                output = softmax_activation(output1.dot(self.weights_output) + self.bias_output)[0]
        else:
            bias = controller[:3].reshape(1, 3)
            weights = controller[3:].reshape((len(inputs), 3))
            output = softmax_activation(inputs.dot(weights) + bias)[0]

        if output[1] > 0.5:
            return 'turn_left'
        elif output[2] > 0.5:
            return 'turn_right'
        else:
            return 'move_forward'
        

def relu_activation(x):
    return np.maximum(0, x)

def softmax_activation(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=1, keepdims=True)

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
        for joint, value in zip(joint_list, temp2):
            readings[joint].append(value)
    sensor_modes = {joint: stats.mode(values)[0][0] for joint, values in readings.items()}

    front_sensors = {
        "FrontC": sensor_modes["FrontC"],
        "FrontR": sensor_modes["FrontR"],
        "FrontL": sensor_modes["FrontL"],
        "FrontRR": sensor_modes["FrontRR"],
        "FrontLL": sensor_modes["FrontLL"]
    }
    
    return front_sensors

def move_forward(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=speed, millis=duration)
    # time.sleep(duration/1000)

def turn_left(rob, speed, duration):
    rob.move_blocking(left_speed=-speed, right_speed=speed, millis=duration)
    # time.sleep(duration/1000)


def turn_right(rob, speed, duration):
    rob.move_blocking(left_speed=speed, right_speed=-speed, millis=duration)
    # time.sleep(duration/1000)




def _digitize_irs(irs):
    # return np.digitize(irs, [10, 100, 250, 300])
    return np.digitize(irs, [250, 275, 285, 300])
    return np.digitize(irs, [100, 250, 300])

def _calc_dist(img, green=True):
    distance = get_dist(img, size=48, green=green)
    # map distance to values from 0 to 3 where 3 means no object
    if distance is None:
        distance = 3
    else:
        distance = np.digitize(distance, [48 // 5, 48 // 2.5])
    return distance
    
def _get_obs(rob):
    irs = rob.read_irs()
    irs_discrete = _digitize_irs(irs)

    img = rob.get_image_front()
    # if step_count == 1:
        # hsv = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        # cv2.imwrite(MODELS_DIR / 'red.jpg', hsv)
    distance_green = _calc_dist(img)
    distance_red = _calc_dist(img, green=False)

    observation = {'irs': irs_discrete, 'dist_g': distance_green, 'dist_r': distance_red}
    return observation


def _calc_metric(rob):
    base_xyz = rob.base_position()
    red_food_xyz = rob.red_food_position()

    xs = (base_xyz.x - red_food_xyz.x)**2
    ys = (base_xyz.y - red_food_xyz.y)**2
    distance = np.sqrt(xs + ys)
    # scale to usable numbers
    return int(distance * 100)

# def fitness(individual, rob, start_position, start_orientation, target_position, controller, steps):
#     rob.set_position(start_position, start_orientation)

#     food_handle = rob._get_childscript(rob._get_object("/Food"))
#     print('food_handle',food_handle)
#     food_position = Position(x=-3, y=0.325, z=0.9)
#     rob._sim.setObjectPosition(34, rob._sim.handle_world, [food_position.x, food_position.y, food_position.z])

#     controller.set(individual, n_inputs=5)
#     collisions = 0
#     stuck = 0
#     distance_to_target = float('inf')
#     threshold = 250  

#     previous_positions = []
#     stuck_threshold = 5  
    
#     captured = False
#     reward = 0

#     # Initial distance to the target position
#     initial_distance = math.sqrt((start_position.x - food_position.x) ** 2 + (start_position.y - food_position.y) ** 2)
#     previous_distance = initial_distance

#     for step in range(steps):
#         sensor_dict = read_irs_sensors(rob)
#         sensor_inputs = list(sensor_dict.values())
#         action = controller.control(sensor_inputs, individual)

#         observation = _get_obs(rob)

#         # print(observation['irs'],'\n', observation['dist_r'])
#         if observation['irs'][4] > 2 and observation['dist_r'] < 2:
#             captured = True
#             reward += 50
#             print('CAPTURED ', captured)

#         elif observation['irs'][4] > 2 or observation['irs'][0] > 2 or observation['irs'][1] > 2:
#             reward += -50
#             print('I hit the wall')

#         elif observation['irs'][3] > 2 and observation['irs'][5] > 2:
#             reward += -50
#             print('I hit the wall')
#         elif observation['irs'][2] > 2 and observation['irs'][7] > 2:
#             reward += -50
#             print('I hit the wall')

#         # if action == 1:
#         #     reward += -5

#         if observation['irs'][4] > 2 and observation['dist_r'] < 2 and observation['dist_g'] < 3:
#             reward += 100
#             print('I see the base ')

#         current_position = rob.get_position()
#         previous_positions.append(current_position)        
#         if len(previous_positions) > stuck_threshold:
#             previous_positions.pop(0)
#         if len(previous_positions) == stuck_threshold:            
#             average_movement = sum(math.sqrt((previous_positions[i].x - previous_positions[i-1].x) ** 2 +(previous_positions[i].y - previous_positions[i-1].y) ** 2)for i in range(1, stuck_threshold)) / (stuck_threshold - 1)            
#             if average_movement < 0.07:
#                 print("stuck")
#                 stuck += 1
#                 previous_positions = [current_position]
#                 reward += -100        
            
#         if rob.base_detects_food():
#             reward += 1000
#             rob.stop_simulation()

#         # Calculate distance to the target position
#         current_distance = math.sqrt((current_position.x - food_position.x) ** 2 + (current_position.y - food_position.y) ** 2)

#         # Reward or penalize based on movement towards the target position
#         if current_distance < previous_distance:
#             reward += 50  # Reward for moving closer
#         else:
#             reward -= 50  # Penalize for moving further away

#         previous_distance = current_distance


#         if action == "move_forward":
#             move_forward(rob, speed=50, duration=500)
#         elif action == "turn_left":
#             turn_left(rob, speed=30, duration=200)
#         elif action == "turn_right":
#             turn_right(rob, speed=30, duration=200)

#     print(reward)

#     return reward

def fitness(individual, rob, start_position, start_orientation, target_position, controller, steps):
    rob.set_position(start_position, start_orientation)

    # food_handle = rob._get_childscript(rob._get_object("/Food"))
    # print('food_handle', food_handle)
    food_position = Position(x=-3, y=0.325, z=0.9)
    rob._sim.setObjectPosition(34, rob._sim.handle_world, [food_position.x, food_position.y, food_position.z])

    controller.set(individual, n_inputs=5)
    collisions = 0
    stuck = 0
    distance_to_food = float('inf')
    distance_to_base = float('inf')
    threshold = 250  

    previous_positions = []
    stuck_threshold = 5  
    
    captured = False
    reward = 0
    rotations = 0

    # Initial distance to the food
    initial_distance_to_food = math.sqrt((start_position.x - food_position.x) ** 2 + (start_position.y - food_position.y) ** 2)
    previous_distance_to_food = initial_distance_to_food
    print(f"Initial distance to food: {initial_distance_to_food}")

    for step in range(steps):
        sensor_dict = read_irs_sensors(rob)
        sensor_inputs = list(sensor_dict.values())
        action = controller.control(sensor_inputs, individual)

        observation = _get_obs(rob)

        current_position = rob.get_position()
        
        # Phase 1: Move towards the food
        if not captured:
            current_distance_to_food = math.sqrt((current_position.x - food_position.x) ** 2 + (current_position.y - food_position.y) ** 2)

            # Reward or penalize based on movement towards the food

            print("current_distance_to_food,previous_distance_to_food")
            print(current_distance_to_food,previous_distance_to_food)

            if current_distance_to_food - previous_distance_to_food > 0.099:
                reward += 50  # Reward for moving significantly closer to food
                print(f"Step {step}: Moved closer to food, reward increased")
            else:
                reward -= 50  # Penalize for not moving closer to food
                print(f"Step {step}: Did not move closer to food, reward decreased")

            if observation['irs'][4] > 2 and observation['dist_r'] < 2:
                captured = True
                reward += 200
                print(f"Step {step}: Food captured, reward increased")
            
            previous_distance_to_food = current_distance_to_food
        
        else:
            # Phase 2: Push the food to the base
            current_distance_to_base = math.sqrt((current_position.x - target_position.x) ** 2 + (current_position.y - target_position.y) ** 2)
            print(f"Step {step}: Current distance to base: {current_distance_to_base}")

            # Reward or penalize based on movement towards the base
            if current_distance_to_base - distance_to_base >0.099:
                reward += 100  # Reward for moving significantly closer to base with food
                print(f"Step {step}: Moved closer to base, reward increased")
            else:
                reward -= 50  # Penalize for not moving closer to base
                print(f"Step {step}: Did not move closer to base, reward decreased")

            distance_to_base = current_distance_to_base

            # Check if the base detects food
            if rob.base_detects_food():
                reward += 1000
                print("Food detected at base, large reward!")
                rob.stop_simulation()
                break

        previous_positions.append(current_position)        
        if len(previous_positions) > stuck_threshold:
            previous_positions.pop(0)
        if len(previous_positions) == stuck_threshold:            
            average_movement = sum(math.sqrt((previous_positions[i].x - previous_positions[i-1].x) ** 2 +(previous_positions[i].y - previous_positions[i-1].y) ** 2) for i in range(1, stuck_threshold)) / (stuck_threshold - 1)            
            if average_movement < 0.07:
                print("stuck")
                stuck += 1
                previous_positions = [current_position]
                reward -= 100  # Penalize for being stuck

        # Execute action
        if action == "move_forward":
            move_forward(rob, speed=50, duration=500)
        elif action == "turn_left":
            turn_left(rob, speed=30, duration=200)
            rotations += 1
        elif action == "turn_right":
            turn_right(rob, speed=30, duration=200)
            rotations += 1
        
        # Penalize for excessive rotations
        if rotations > 10:
            reward -= 50
            print(f"Step {step}: Excessive rotation penalized")

    print(f"Final reward: {reward}")

    return reward




def initialize_population(size, n_weights):
    return [np.random.uniform(-1, 1, n_weights) for _ in range(size)]

def tournament_selection(population, fitnesses, num_parents, tournament_size=5):
    selected_parents = []
    for _ in range(num_parents):
        # Sample a tournament group
        tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
        # Sort by absolute fitness value
        tournament.sort(key=lambda x: abs(x[1]))
        # Select the best individual from the tournament
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
        best_fitness = min(fitnesses, key=lambda x: abs(x))  
        checkpoint = {
            'generation': generation,
            'fitnesses': fitnesses,
            'best_fitness': best_fitness,
            'best_individual': best_individual.tolist(),
            'population': [ind.tolist() for ind in population]
        }
        with open(filename, 'a') as f:
            f.write(json.dumps(checkpoint) + '\n')
        print(f"Checkpoint saved successfully at generation {generation}")
    except Exception as e:
        print(f"Error saving checkpoint: {e}")

def load_checkpoint_jsonl(filename):
    try:
        with open(filename, 'r') as f:
            last_line = None
            for last_line in f:
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

checkpoint_path = str(RESULT_DIR / "checkpoint_task_3_v3.jsonl")
n_hidden_layers = [32, 16] 
n_inputs = 5
n_outputs = 3

def evolutionary_algorithm(rob, start_position, start_orientation, target_position,
                           generations=25, population_size=10,
                           checkpoint_file=checkpoint_path, continue_from_checkpoint=True,
                           steps=20):

    n_weights = n_inputs * n_hidden_layers[0] + n_hidden_layers[0] + n_hidden_layers[0] * n_hidden_layers[1] + n_hidden_layers[1] + n_hidden_layers[1] * n_outputs  + n_outputs 

    controller = robot_controller(n_hidden_layers)
    
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
        
        num_parents = max(2, population_size // 10)
        parents = tournament_selection(population, fitnesses, num_parents, tournament_size=5)

        new_population = []
        while len(new_population) < (population_size - num_parents):
            parent1, parent2 = random.sample(parents, 2)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(mutate(child1))
            new_population.append(mutate(child2))

        population = new_population[:population_size - num_parents] + parents

        best_individual_index = fitnesses.index(min(fitnesses, key=lambda x: abs(x)))
        best_individual = population[best_individual_index]
        save_checkpoint_jsonl(population, fitnesses, best_individual, generation, checkpoint_file)

        best_fitness = min(fitnesses, key=lambda x: abs(x))
        print(f"Generation {generation}: Best Fitness = {best_fitness}")



########
# For testing


def load_best_individual():
    try:
        best_individual = None
        best_fitness_diff = float('inf')
        
        with open(checkpoint_path, 'r') as f:
            for line in f:
                checkpoint = json.loads(line)
                fitness_diff = abs(checkpoint['best_fitness'])
                if fitness_diff < best_fitness_diff:
                    best_fitness_diff = fitness_diff
                    best_individual = np.array(checkpoint['best_individual'])
                    
        if best_individual is None:
            raise ValueError("No valid best individual found in checkpoint file")

        print(f"Best individual loaded successfully with fitness closest to zero: {best_fitness_diff}")
        return best_individual

    except Exception as e:
        print(f"Error loading best individual: {e}")
        return None



def test_best_individual(rob, start_position, start_orientation, target_position, steps=30):



    food_handle = rob._get_childscript(rob._get_object("/Food"))
    position = Position(x=-3, y=0.325, z=0.9)
    rob._sim.setObjectPosition(34, rob._sim.handle_world, [position.x, position.y, position.z])


    controller = robot_controller(n_hidden_layers)
    best_individual = load_best_individual()
    controller.set(best_individual, n_inputs)
    collisions = 0
    sensor_data_per_step = []
    threshold = 250

    
    speed=50
    duration=500

    for _ in range(steps):
        sensor_dict = read_irs_sensors(rob)
        sensor_data_per_step.append(sensor_dict.copy())  
        sensor_inputs = list(sensor_dict.values())
        action = controller.control(sensor_inputs, best_individual)


        
        print('action', action)

        if action == "move_forward":
            move_forward(rob, speed, duration)
            # time.sleep(duration/1000)

        elif action == "turn_left":
            turn_left(rob, speed, duration)
            # time.sleep(duration/1000)
        elif action == "turn_right":
            turn_right(rob, speed, duration)
            # time.sleep(duration/1000)

        # sensor_dict = read_irs_sensors(rob)
        # if (sensor_dict["FrontC"] > threshold or
        #     sensor_dict["FrontR"] > threshold or
        #     sensor_dict["FrontL"] > threshold):
        #     collisions += 1

    return sensor_data_per_step