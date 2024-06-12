import numpy as np
import math
import os
import shutil

def rotation_matrix_to_euler_angles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    
    singular = sy < 1e-6

    if not singular:
        yaw = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        roll = math.atan2(R[1, 0], R[0, 0])
    else:
        yaw = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        roll = 0

    return yaw, pitch, roll

def parse_and_find_max_angles(filename):
    max_pitch = -float('inf')
    max_yaw = -float('inf')
    all_points = []
    line_numbers = []

    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if len(parts) < 6:
                # Skip the line if it doesn't have enough parts
                continue
            
            normmat_str = parts[-1]
            try:
                normmat_vals = list(map(float, normmat_str.split(',')))
            except ValueError:
                # Skip the line if normmat_str can't be converted to floats
                continue
            
            if len(normmat_vals) != 9:
                # Skip the line if normmat doesn't have 9 values
                continue
            
            R = np.array(normmat_vals).reshape(3, 3)
            
            yaw, pitch, _ = rotation_matrix_to_euler_angles(R)
            
            all_points.append((yaw, pitch))
            line_numbers.append(line_number)
            
            max_pitch = max(max_pitch, pitch)
            max_yaw = max(max_yaw, yaw)

    return max_pitch, max_yaw, all_points, line_numbers

def find_closest_point(target, points):
    closest_index = min(range(len(points)), key=lambda i: (points[i][0] - target[0])**2 + (points[i][1] - target[1])**2)
    return closest_index

def find_largest_rectangle_points(points):
    yaws, pitches = zip(*points)
    
    min_yaw = min(yaws)
    max_yaw = max(yaws)
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    
    origin = (0, 0)
    top_left = (min_yaw, max_pitch)
    top_right = (max_yaw, max_pitch)
    bottom_left = (min_yaw, min_pitch)
    bottom_right = (max_yaw, min_pitch)
    
    mid_top = ((min_yaw + max_yaw) / 2, max_pitch)
    mid_bottom = ((min_yaw + max_yaw) / 2, min_pitch)
    mid_left = (min_yaw, (min_pitch + max_pitch) / 2)
    mid_right = (max_yaw, (min_pitch + max_pitch) / 2)
    
    return [origin, top_left, top_right, bottom_left, bottom_right, mid_top, mid_bottom, mid_left, mid_right]

def interpolate_grid_points(points, grid_size):
    yaws, pitches = zip(*points)
    
    min_yaw = min(yaws)
    max_yaw = max(yaws)
    min_pitch = min(pitches)
    max_pitch = max(pitches)
    
    yaw_steps = np.linspace(min_yaw, max_yaw, int(math.sqrt(grid_size)))
    pitch_steps = np.linspace(min_pitch, max_pitch, int(math.sqrt(grid_size)))
    
    grid_points = []
    for yaw in yaw_steps:
        for pitch in pitch_steps:
            grid_points.append((yaw, pitch))
    
    # Ensure we only keep the required number of points
    grid_points = sorted(grid_points, key=lambda point: (point[0], point[1]))[:grid_size]
    
    return grid_points

def move_images_to_folder(filename, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if len(parts) < 6:
                # Skip the line if it doesn't have enough parts
                continue
            
            image_path = parts[0]  # Assuming image path is in the first column
            image_name = os.path.basename(image_path)
            destination_path = os.path.join(destination_folder, image_name)
            complete_image_path = "/path/to/data/folder" + image_path #ADD PATH HERE

            if os.path.exists(complete_image_path):
                shutil.copy(complete_image_path, destination_path)
                print(f"Moved '{image_name}' to '{destination_folder}'")
            else:
                print(f"Warning: '{image_name}' not found at '{image_path}'")


def main():
    filename = '/mntdata/Atishna/train.label'
    destination_folder = '/mntdata/Atishna/pointsele03' #ADD PATH HERE

    _, _, points, line_numbers = parse_and_find_max_angles(filename)

    if not points:
        print("No points were extracted from the file.")
        return

    rectangle_points = find_largest_rectangle_points(points)

    grid_5_points = rectangle_points[:5]  # Origin + 4 corners
    grid_9_points = rectangle_points      # Origin + 4 corners + 4 midpoints
    grid_16_points = interpolate_grid_points(rectangle_points, 16)
    grid_25_points = interpolate_grid_points(rectangle_points, 25)

    def print_points(grid_points, grid_name):
        print(f'{grid_name} Points (yaw, pitch):')
        for point in grid_points:
            closest_index = find_closest_point(point, points)
            print(f'Radians: {points[closest_index]}, Degrees: ({math.degrees(points[closest_index][0])}, {math.degrees(points[closest_index][1])}), Line: {line_numbers[closest_index]}')
        print()

    print_points(grid_5_points, '5 Grid')
    print_points(grid_9_points, '9 Grid')
    print_points(grid_16_points, '16 Grid')
    print_points(grid_25_points, '25 Grid')

    move_images_to_folder(filename, destination_folder)


if __name__ == '__main__':
    main()



