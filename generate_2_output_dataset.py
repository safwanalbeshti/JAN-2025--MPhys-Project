import numpy as np
import matplotlib.pyplot as plt
import random

# Define the robot and environment parameters
robot_width = 15
robot_height = 10
robot_position = (50, 5)  # Center of the robot
sensor_angles = [-25, -15, -5, 5, 15, 25]  # degrees, distributed around zero and upwards
sensor_range = 100
obstacle_size_range = (10, 30)
threshold_distance = 40  # constant minimum distance for deviation decision
distance_normalization_factor = 20


# Function to generate a single random obstacle within the environment size
def generate_obstacle(obstacle_size_range, environment_size):
    size = random.randint(*obstacle_size_range)
    x = random.randint(0, environment_size[0] - size)  # constrain x to fit within environment
    y = random.randint(10, environment_size[1] - size)  # constrain y to fit within environment
    angle = random.uniform(0, 360)  # random tilt angle
    return (x, y, size, size, angle)


# Function to calculate distances to the obstacle from sensor angles
def calculate_distances(robot_position, sensor_angles, obstacle, sensor_range):
    distances = []
    obs_x, obs_y, obs_w, obs_h, obs_angle = obstacle

    robot_center_x = robot_position[0]
    robot_center_y = robot_position[1]

    for angle in sensor_angles:
        print(angle)
        rad = np.deg2rad(angle)

        # Calculate the start point of the sensor line (center of the robot)
        start_x = robot_center_x
        start_y = robot_center_y

        # Check if sensor line intersects with any side of the obstacle
        intersected = False

        # Calculate intersection point with left side of the obstacle
        if angle > 0:
            print("angle > 0")
            if rad != 0:  # Avoid division by zero for vertical sensor lines
                x_inter_left = obs_x
                y_inter_left = (x_inter_left - start_x) / np.sin(rad) * np.cos(rad) + start_y

                if obs_y <= y_inter_left <= obs_y + obs_h:
                    intersected = True
                    distance = np.sqrt((robot_center_x - x_inter_left) ** 2 + (robot_center_y - y_inter_left) ** 2)
                    distances.append(round(distance / distance_normalization_factor) + 1)
            print(distances)

        # Calculate intersection point with right side of the obstacle
        if angle < 0:
            print("angle < 0")
            if rad != 0:  # Avoid division by zero for vertical sensor lines
                x_inter_right = obs_x + obs_w
                y_inter_right = (x_inter_right - start_x) / np.sin(rad) * np.cos(
                    rad) + start_y  # <- abs position of bottom-right corner of obstacle

                if obs_y <= y_inter_right <= obs_y + obs_h:
                    intersected = True
                    distance = np.sqrt((robot_center_x - x_inter_right) ** 2 + (robot_center_y - y_inter_right) ** 2)
                    distances.append(round(distance / distance_normalization_factor) + 1)
            print(distances)

        # Calculate intersection point with lower side of the obstacle
        if np.abs(angle) <= 90:
            print("|angle| < 90")
            if rad != np.pi / 2 and rad != -np.pi / 2:  # Avoid division by zero for horizontal sensor lines
                y_inter_lower = obs_y
                x_inter_lower = (y_inter_lower - start_y) / np.cos(rad) * np.sin(rad) + start_x

                if obs_x <= x_inter_lower <= obs_x + obs_w:
                    intersected = True
                    distance = np.sqrt((robot_center_x - x_inter_lower) ** 2 + (robot_center_y - y_inter_lower) ** 2)
                    distances.append(round(distance / distance_normalization_factor) + 1)
            print(distances)

        if not intersected:
            distances.append(round(sensor_range / distance_normalization_factor) + 1)  # Max distance if no intersection

    return distances


# Function to determine deviation based on minimum distance
def determine_deviation(distances, threshold):
    return 1 if any(d <= (threshold / distance_normalization_factor) + 1 for d in distances) else 0


# Function to plot the environment, robot, and path
def plot_scenario(robot_position, obstacle, distances, deviation):
    fig, ax = plt.subplots()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Draw robot
    robot_rect = plt.Rectangle((robot_position[0] - robot_width / 2, robot_position[1] - robot_height / 2),
                               robot_width, robot_height, color='lightblue')
    ax.add_patch(robot_rect)

    # Draw obstacle
    obs_x, obs_y, obs_w, obs_h, obs_angle = obstacle
    obstacle_rect = plt.Rectangle((obs_x, obs_y), obs_w, obs_h, color='black')
    ax.add_patch(obstacle_rect)

    # Draw sensor lines
    for angle, distance in zip(sensor_angles, distances):
        rad = np.deg2rad(angle)
        start_x = robot_position[0]
        start_y = robot_position[1]

        # Calculate the end point of the sensor line
        end_x = start_x + (distance - 1) * distance_normalization_factor * np.sin(rad)
        end_y = start_y + (distance - 1) * distance_normalization_factor * np.cos(rad)

        ax.plot([start_x, end_x],
                [start_y, end_y], 'k--')

    # Draw path
    if deviation == 1:
        path = plt.Arrow(robot_position[0], robot_position[1] + robot_height / 2, 0, -10, width=10, color='red')
    else:
        path = plt.Arrow(robot_position[0], robot_position[1] + robot_height / 2, 0, 50, width=5, color='green')
    ax.add_patch(path)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


# Function to generate data and visualize scenarios

def generate_data(num_samples, environment_size):
    inputs = []
    outputs = []
    count_0 = 0
    count_1 = 0
    target_per_class = num_samples // 2

    while len(inputs) < num_samples:
        obstacle = generate_obstacle(obstacle_size_range, environment_size)
        distances = calculate_distances(robot_position, sensor_angles, obstacle, sensor_range)
        deviation = determine_deviation(distances, threshold_distance)

        if deviation == 0 and count_0 < target_per_class:
            count_0 += 1
        elif deviation == 1 and count_1 < target_per_class:
            count_1 += 1
        else:
            continue

        plot_scenario(robot_position, obstacle, distances, deviation)
        inputs.append(distances)
        outputs.append(deviation)
    return [inputs, outputs]


# Functions to generate training and testing data
def generate_train_data(num_samples, environment_size):
    print("Generating training data...")
    train_data = generate_data(num_samples, environment_size)
    for distances, deviation in zip(train_data[0], train_data[1]):
        print(f"Input (distances): {distances} -> Output (deviation): {deviation}")
    return train_data


def generate_test_data(num_samples, environment_size):
    print("Generating testing data...")
    test_data = generate_data(num_samples, environment_size)
    for distances, deviation in zip(test_data[0], test_data[1]):
        print(f"Input (distances): {distances} -> Output (deviation): {deviation}")
    return test_data

# Example usage
environment_size = (100, 100)  # Define the environment size
DatasTrain = generate_train_data(10, environment_size)

for i in range(len(DatasTrain[0])):
    print(DatasTrain[0][i], DatasTrain[1][i])

DatasTest= generate_test_data(20, environment_size)
