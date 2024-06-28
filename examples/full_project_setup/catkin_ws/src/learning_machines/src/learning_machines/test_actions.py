import cv2

from data_files import FIGRURES_DIR,RESULT_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
    HardwareRobobo,
    Position,
    Orientation,
)
import json



from learning_machines import task_0
from learning_machines import task_1
from learning_machines import task_1_v2
from learning_machines import task_3


def test_emotions(rob: IRobobo):
    rob.set_emotion(Emotion.HAPPY)
    rob.talk("Hello")
    rob.play_emotion_sound(SoundEmotion.PURR)
    rob.set_led(LedId.FRONTCENTER, LedColor.GREEN)


def test_move_and_wheel_reset(rob: IRobobo):
    rob.move_blocking(100, 100, 1000)
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(1)
    print("after reset: ", rob.read_wheels())



def test_move_and_wheel_reset22(rob: IRobobo):
    rob.move_blocking(50000, 100, 1000)
    print("\n\n\n\n#############\n\n")
    print("before reset: ", rob.read_wheels())
    rob.reset_wheels()
    rob.sleep(2)
    print("after reset: ", rob.read_wheels())


def test_sensors(rob: IRobobo):
    print("IRS data: ", rob.read_irs())
    image = rob.get_image_front()
    cv2.imwrite(str(FIGRURES_DIR / "photo.png"), image)
    print("Phone pan: ", rob.read_phone_pan())
    print("Phone tilt: ", rob.read_phone_tilt())
    print("Current acceleration: ", rob.read_accel())
    print("Current orientation: ", rob.read_orientation())


def test_phone_movement(rob: IRobobo):
    rob.set_phone_pan_blocking(20, 100)
    print("Phone pan after move to 20: ", rob.read_phone_pan())
    rob.set_phone_tilt_blocking(50, 100)
    print("Phone tilt after move to 50: ", rob.read_phone_tilt())


def test_sim(rob: SimulationRobobo):
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.stop_simulation()
    print(rob.get_sim_time())
    print(rob.is_running())
    rob.play_simulation()
    print(rob.get_sim_time())
    print(rob.get_position())


def test_hardware(rob: HardwareRobobo):
    print("Phone battery level: ", rob.read_phone_battery())
    print("Robot battery level: ", rob.read_robot_battery())


# def run_all_actions(rob: IRobobo):
#     if isinstance(rob, SimulationRobobo):
#         rob.play_simulation()
#     test_emotions(rob)
#     test_sensors(rob)
#     test_move_and_wheel_reset(rob)

#     test_move_and_wheel_reset22(rob)


#     if isinstance(rob, SimulationRobobo):
#         test_sim(rob)

#     if isinstance(rob, HardwareRobobo):
#         test_hardware(rob)

#     test_phone_movement(rob)

#     if isinstance(rob, SimulationRobobo):
#         rob.stop_simulation()





# def train_task_1(rob):
#     start_position = Position(x=0.0, y=0.0, z=0.09)  # Set the starting position
#     start_orientation = Orientation(yaw=-175.00036138789557, pitch=-19.996487020842473, roll=4.820286812070959e-05)  # Set the starting orientation
#     target_position = Position(x=-1, y=1.5, z=0.0)  # Set the target position

#     try:
#         task_1_v2.evolutionary_algorithm(rob, start_position, start_orientation, target_position)
#         print("Done training")
#     finally:
#         rob.stop_simulation()  # Stop the simulation when done



sensor_data_sim_path = str(RESULT_DIR / "sensor_data_simulation_new.jsonl")
sensor_data_hardware_path = str(RESULT_DIR / "sensor_data_hardware_new.jsonl")

def save_sensor_data(sensor_data, filename):
    with open(filename, 'a') as f:
        for data in sensor_data:
            f.write(json.dumps(data) + '\n')
    print(f"Sensor data saved successfully to {filename}")

# def test_task_1(rob, sensor_data_path):
#     start_position = Position(x=0.0, y=0.0, z=0.09)  # Set the starting position
#     start_orientation = Orientation(yaw=-175.00036138789557, pitch=-19.996487020842473, roll=4.820286812070959e-05)  # Set the starting orientation
#     target_position = Position(x=-0.5, y=1, z=0.0)  # Set the target position
#     sensor_data_per_step = task_1_v2.test_best_individual(rob, start_position, start_orientation, target_position)
#     save_sensor_data(sensor_data_per_step, sensor_data_path)
#     print("Done testing")




def train_task_3(rob):
    start_position = Position(x=-2.4, y=0.077, z=0.09)  # Set the starting position
    start_orientation = Orientation(yaw=-175.00036138789557, pitch=-19.996487020842473, roll=4.820286812070959e-05)  # Set the starting orientation
    target_position = Position(x=-3.500, y=1.175, z=0.002)  # Set the target position

    try:
        task_3.evolutionary_algorithm(rob, start_position, start_orientation, target_position)
        print("Done training")
    finally:
        rob.stop_simulation()  # Stop the simulation when done


def test_task_3(rob, sensor_data_path):
    start_position = Position(x=-2.4, y=0.077, z=0.09)  # Set the starting position
    start_orientation = Orientation(yaw=-175.00036138789557, pitch=-19.996487020842473, roll=4.820286812070959e-05)  # Set the starting orientation
    target_position = Position(x=-3.500, y=1.175, z=0.002)  # Set the target position
    sensor_data_per_step = task_3.test_best_individual(rob, start_position, start_orientation, target_position)
    # save_sensor_data(sensor_data_per_step, sensor_data_path)
    print("Done testing")

def run_all_actions(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()        
        train_task_3(rob)
        # test_task_3(rob,sensor_data_sim_path)

    # task_0.avoid_obstacle(rob)
    # task_0.touch_wall_backup(rob)


    
    


    # if isinstance(rob, SimulationRobobo):
    #     test_sim(rob)

    if isinstance(rob, HardwareRobobo):
        test_hardware(rob)

        # test_task_1(rob,sensor_data_hardware_path)

    # test_phone_movement(rob)

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()



# if __name__ == "__main__":
#     robobo = SimulationRobobo()  # Initialize your simulation robot
#     robobo.play_simulation()  # Start the simulation

#     target_position = Position(x=10.0, y=10.0, z=0.0)  # Set the target position
#     try:
#         best_path = evolutionary_algorithm(robobo, target_position)
#         print("Best Path:", best_path)
#     finally:
#         robobo.stop_simulation()  # Stop the simulation when done
