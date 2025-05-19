#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from openflexure_microscope_client import MicroscopeClient
from ultralytics import YOLO



# Load the malaria detection model
model = YOLO(r"C:\Users\Ezekiel\Desktop\model trained\MALARIA FINAL MODEL\best.pt")  # Update model path

# Set the output directory
desktop = Path.home() / "Desktop"
output_dir = desktop / "test 3" / "c"
os.makedirs(output_dir, exist_ok=True)

# File to save statistics
stats_file = os.path.join(output_dir, "statistics.txt")

# Initialize total counts and statistics storage
total_infected = 0
total_non_infected = 0
total_wbc = 0
total_time = 0
image_count = 0
statistics = []

# Store the starting position
starting_pos = microscope.position

# Step sizes and scanning parameters
step_size_x = 400
step_size_y = 400
step_size_z = 50
evaluation_steps = 4
fine_focus_steps = 10
num_x_steps = 10
num_y_steps = 10
scan_direction = 1

# Function to compute Laplacian variance (focus measurement)
def compute_laplacian_variance(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()

# Function to move the microscope
def move_microscope(x=None, y=None, z=None):
    pos = microscope.position
    if x is not None:
        pos['x'] = x
    if y is not None:
        pos['y'] = y
    if z is not None:
        pos['z'] = z
    microscope.move(pos)

# Function to evaluate focus quality
def evaluate_focus(direction, steps, initial_z):
    variances = []
    for j in range(1, steps + 1):
        new_z = initial_z + (j * step_size_z * direction)
        move_microscope(z=new_z)
        time.sleep(0.2)
        img = microscope.grab_image()
        img_np = np.array(img)
        variance = compute_laplacian_variance(img_np)
        variances.append((variance, img, new_z))
    return variances

# Function to find the best focus
def find_best_focus(variances, best_variance, best_img, best_z):
    for variance, img, z in variances:
        if variance > best_variance:
            best_variance, best_img, best_z = variance, img, z
    return best_variance, best_img, best_z

# Start scanning process
for y_step in range(num_y_steps):
    for x_step in range(num_x_steps):
        start_time = time.time()

        # Move in X-direction
        new_x = microscope.position['x'] + (scan_direction * step_size_x)
        move_microscope(x=new_x)
        time.sleep(0.5)

        # Get initial Z position and capture image for focus evaluation
        initial_z = microscope.position['z']
        initial_img = microscope.grab_image()
        initial_img_np = np.array(initial_img)
        initial_variance = compute_laplacian_variance(initial_img_np)
        best_focus_image, best_variance, best_z = initial_img, initial_variance, initial_z

        # Evaluate focus downward and upward
        downward_variances = evaluate_focus(-1, evaluation_steps, initial_z)
        upward_variances = evaluate_focus(1, evaluation_steps, initial_z)
        best_variance, best_focus_image, best_z = find_best_focus(downward_variances, best_variance, best_focus_image, best_z)
        best_variance, best_focus_image, best_z = find_best_focus(upward_variances, best_variance, best_focus_image, best_z)

        # Fine focus
        optimal_direction = 1 if best_z > initial_z else -1
        fine_focus_variances = evaluate_focus(optimal_direction, fine_focus_steps, best_z)
        best_variance, best_focus_image, best_z = find_best_focus(fine_focus_variances, best_variance, best_focus_image, best_z)

        # Save best-focused image
        image_count += 1
        image_path = os.path.join(output_dir, f"image_{image_count}.png")
        best_focus_image.save(image_path)
        move_microscope(z=best_z)  # Ensure we return to best-focused Z

        # Run YOLO model on the best-focused image
        results = model(np.array(best_focus_image), conf=0.25)
        predictions = results[0]
        class_names = model.names  # Dictionary of class labels

        # Save detected image (using plot() to get the annotated image)
        predicted_image = results[0].plot()  
        predicted_image_pil = Image.fromarray(predicted_image).convert("RGB")
        predicted_image_path = os.path.join(output_dir, f"image_{image_count}_predicted.png")
        predicted_image_pil.save(predicted_image_path)

        # Extract statistics
        infected_count = sum(1 for c in predictions.boxes.cls if class_names[int(c)].lower() == "infected rbc")
        non_infected_count = sum(1 for c in predictions.boxes.cls if class_names[int(c)].lower() in ["non infected rbc", "non imfected rbc"])
        wbc_count = sum(1 for c in predictions.boxes.cls if class_names[int(c)].lower() == "wbc")

        total_infected += infected_count
        total_non_infected += non_infected_count
        total_wbc += wbc_count

        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        total_time += time_taken

        statistics.append(f"""Image {image_count}:
  - Infected RBCs: {infected_count}
  - Non-infected RBCs: {non_infected_count}
  - WBCs: {wbc_count}
  - Processing Time: {time_taken} seconds
----------------------------------------""")

        print(f"Image {image_count}: Infected RBCs = {infected_count}, Non-infected RBCs = {non_infected_count}, WBCs = {wbc_count}, Time taken = {time_taken}s")

        # Return to the best-focused Z position before moving to the next position
        move_microscope(z=best_z)

        if image_count >= 100:  # Stop after 100 images (for testing, you can set a lower number)
            break
    if image_count >= 100:
        break

    # Move down for the next row and reverse horizontal direction
    new_y = microscope.position['y'] + step_size_y
    move_microscope(y=new_y)
    time.sleep(0.5)
    scan_direction *= -1

# Move microscope back to starting position
microscope.move(starting_pos)

# Compute average processing time
average_time = round(total_time / image_count, 2)

# Save statistics to file
with open(stats_file, "w") as f:
    f.write("Malaria Detection Statistics\n")
    f.write("=" * 40 + "\n")
    f.write("\n".join(statistics))
    f.write(f"\nTotal Statistics:\n  - Total Infected RBCs: {total_infected}\n  - Total Non-Infected RBCs: {total_non_infected}\n  - Total WBCs: {total_wbc}\n  - Total Processing Time: {total_time} seconds\n  - Average Processing Time per Image: {average_time} seconds\n")
    f.write("=" * 40 + "\n")

print(f"✅ All images and statistics saved in {output_dir}")

