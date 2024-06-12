import numpy as np
import math
import os
import shutil
from scipy.stats import qmc

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
    all_points = []
    line_numbers = []

    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            parts = line.strip().split()
            if len(parts) < 6:
                continue
            
            normmat_str = parts[-1]
            try:
                normmat_vals = list(map(float, normmat_str.split(',')))
            except ValueError:
                continue
            
            if len(normmat_vals) != 9:
                continue
            
            R = np.array(normmat_vals).reshape(3, 3)
            yaw, pitch, _ = rotation_matrix_to_euler_angles(R)
            all_points.append((yaw, pitch))
            line_numbers.append(line_number)

    return all_points, line_numbers

def halton_sequence_points(bounds, num_points):
    sampler = qmc.Halton(d=2, scramble=False)
    sample = sampler.random(n=num_points)
    lower_bounds, upper_bounds = bounds
    scaled_sample = qmc.scale(sample, lower_bounds, upper_bounds)
    return scaled_sample

def move_images_to_folder(filename, base_path, destination_folder, selected_lines):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    with open(filename, 'r') as file:
        for line_number, line in enumerate(file, start=1):
            if line_number in selected_lines:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                
                image_path = parts[0]
                image_name = os.path.basename(image_path)
                destination_path = os.path.join(destination_folder, image_name)
                complete_image_path = os.path.join(base_path, image_path)

                if os.path.exists(complete_image_path):
                    shutil.copy(complete_image_path, destination_path)
                    print(f"Moved '{image_name}' to '{destination_folder}'")
                else:
                    print(f"Warning: '{image_name}' not found at '{complete_image_path}'")

def main():
    filename = '/mntdata/Atishna/subject_train/Label/train.label'
    base_path = '/mntdata/Atishna/subject_train/Image/train/'

    points, line_numbers = parse_and_find_max_angles(filename)

    if not points:
        print("No points were extracted from the file.")
        return

    yaws, pitches = zip(*points)
    bounds = [(min(yaws), min(pitches)), (max(yaws), max(pitches))]

    num_points_list = [5, 9, 16, 25, 50, 100]

    for num_points in num_points_list:
        halton_points = halton_sequence_points(bounds, num_points)
        print(f"\nHalton Points for {num_points} clusters: {halton_points}")

        selected_lines = []
        selected_points = set()

        for point in halton_points:
            closest_index = find_closest_point(point, points)
            while line_numbers[closest_index] in selected_lines:
                closest_index += 1
                if closest_index >= len(points):
                    closest_index = 0
            selected_lines.append(line_numbers[closest_index])
            selected_points.add(points[closest_index])

        print(f'\nSelected {num_points} Points (yaw, pitch):')
        for point, line_num in zip(halton_points, selected_lines):
            yaw_pitch = points[line_numbers.index(line_num)]
            print(f'Point: {point}, Radians: {yaw_pitch}, Degrees: ({math.degrees(yaw_pitch[0])}, {math.degrees(yaw_pitch[1])}), Line: {line_num}')

        destination_folder = f'points_halton/subject0012_{num_points}'
        move_images_to_folder(filename, base_path, destination_folder, selected_lines)

def find_closest_point(target, points):
    closest_index = min(range(len(points)), key=lambda i: (points[i][0] - target[0])**2 + (points[i][1] - target[1])**2)
    return closest_index

if __name__ == '__main__':
    main()
