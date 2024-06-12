import numpy as np
import math
import os
import shutil
from sklearn.cluster import KMeans

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

def kmeans_select_points(points, num_points):
    kmeans = KMeans(n_clusters=num_points, random_state=0).fit(points)
    return kmeans.cluster_centers_

def move_images_to_folder(filename, destination_folder, line_numbers, selected_lines):
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
                complete_image_path = "/mntdata/Atishna/subject_train/Image/train/" + image_path

                if os.path.exists(complete_image_path):
                    shutil.copy(complete_image_path, destination_path)
                    print(f"Moved '{image_name}' to '{destination_folder}'")
                else:
                    print(f"Warning: '{image_name}' not found at '{image_path}'")

def main():
    filename = '/mntdata/Atishna/subject_train/Label/train.label'

    points, line_numbers = parse_and_find_max_angles(filename)

    if not points:
        print("No points were extracted from the file.")
        return

    num_points_list = [5, 9, 16, 25, 50, 100]

    for num_points in num_points_list:
        selected_points = kmeans_select_points(points, num_points)
        selected_lines = [line_numbers[find_closest_point(point, points)] for point in selected_points]

        print(f'\nSelected {num_points} Points (yaw, pitch):')
        for point, line_num in zip(selected_points, selected_lines):
            print(f'Radians: {points[line_numbers.index(line_num)]}, Degrees: ({math.degrees(points[line_numbers.index(line_num)][0])}, {math.degrees(points[line_numbers.index(line_num)][1])}), Line: {line_num}')

        destination_folder = f'points_clustering/subject0012_{num_points}'
        move_images_to_folder(filename, destination_folder, line_numbers, selected_lines)

def find_closest_point(target, points):
    closest_index = min(range(len(points)), key=lambda i: (points[i][0] - target[0])**2 + (points[i][1] - target[1])**2)
    return closest_index

if __name__ == '__main__':
    main()
