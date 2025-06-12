```python
import time
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from openflexure_microscope_client import MicroscopeClient

# Connect to microscope
microscope = MicroscopeClient("192.168.0.111")

# Parameters
coarse_step_size = 200
coarse_num_steps = 10
fine_step_size = 50
settle_time = 0.01  # Reduced settle time for speed
coarse_drop_threshold = 10


def compute_laplacian_variance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def move_microscope_rel(z_delta):
    microscope.move_rel({"x": 0, "y": 0, "z": z_delta})
    time.sleep(settle_time)


def move_microscope_abs(z_target):
    microscope.move({"x": microscope.position['x'], "y": microscope.position['y'], "z": z_target})
    time.sleep(settle_time)


def capture_image():
    return np.array(microscope.grab_image())


def evaluate_focus_direction(step_size, num_steps):
    base_z = microscope.position['z']
    variances = []

    for _ in range(num_steps):
        move_microscope_rel(step_size)
        var = compute_laplacian_variance(capture_image())
        variances.append((var, microscope.position['z']))
    move_microscope_abs(base_z)

    for _ in range(num_steps):
        move_microscope_rel(-step_size)
        var = compute_laplacian_variance(capture_image())
        variances.append((var, microscope.position['z']))
    move_microscope_abs(base_z)

    best_var, best_z = max(variances, key=lambda x: x[0])
    direction = 1 if best_z > base_z else -1
    return direction


def move_until_drop(step_size, direction):
    positions = []
    variances = []
    images = []

    while True:
        move_microscope_rel(direction * step_size)
        image = capture_image()
        var = compute_laplacian_variance(image)

        positions.append(microscope.position['z'])
        variances.append(var)
        images.append(image)

        if len(variances) >= 3:
            if variances[-2] > variances[-1] and (variances[-2] - variances[-1]) > coarse_drop_threshold:
                break

    max_idx = np.argmax(variances)
    best_z = positions[max_idx]
    best_var = variances[max_idx]
    best_image = images[max_idx]

    move_microscope_abs(best_z)
    return best_z, best_var, best_image, positions, variances


def coarse_focus():
    print("Evaluating direction for coarse focus...")
    direction = evaluate_focus_direction(coarse_step_size, coarse_num_steps)
    print(f"Coarse focus direction: {'up' if direction == 1 else 'down'}")
    best_z, best_var, best_image, z_positions, variances = move_until_drop(coarse_step_size, direction)
    print(f"Coarse focus at Z={best_z} with variance={best_var:.2f}")
    return best_z, best_var, best_image, z_positions, variances


def fine_focus(coarse_z, z_positions, variances):
    print("Starting fine focus sweep...")

    start_z = coarse_z - 200
    end_z = coarse_z + 200

    best_var = -1
    best_image = None
    best_z = None

    for z in range(start_z, end_z + 1, fine_step_size):
        move_microscope_abs(z)
        image = capture_image()
        var = compute_laplacian_variance(image)
        print(f"Z={z}, Variance={var:.4f}")

        if var > best_var:
            best_var = var
            best_image = image
            best_z = z

    print(f"Fine focus complete at Z={best_z} with variance={best_var:.2f}")
    move_microscope_abs(best_z)
    return best_z, best_var, best_image


def autofocus():
    coarse_z, _, coarse_image, z_positions, variances = coarse_focus()
    fine_z, _, fine_image = fine_focus(coarse_z, z_positions, variances)
    return fine_z, fine_image


# Output directory
desktop = Path.home() / "Desktop"
output_dir = desktop / "microscope1" / "405nm testing" / "edith4"
output_dir.mkdir(parents=True, exist_ok=True)

# Store initial position
starting_pos = microscope.position.copy()

# Raster scan parameters
step_size_x = 800
step_size_y = 800
x_direction = 1
x_steps = 0
max_x_steps = 10

# Raster scan and capture 50 images
for i in range(50):
    if i > 0:
        if x_steps < max_x_steps - 1:
            starting_pos['x'] += x_direction * step_size_x
            x_steps += 1
        else:
            starting_pos['y'] += step_size_y
            x_direction *= -1
            x_steps = 0

        print(f"Moving to next position: X={starting_pos['x']}, Y={starting_pos['y']}")
        microscope.move(starting_pos)
        print(f"Position after move: {microscope.position}")

    # Autofocus at current position
    best_z, focused_image = autofocus()

    # Update Z
    starting_pos['z'] = best_z

    # Save focused image
    image_filename = output_dir / f"image_{i+1}.png"
    Image.fromarray(focused_image).save(image_filename)
    print(f"Saved image {i+1} at {image_filename}")

    # Optional: Preview
    plt.imshow(focused_image)
    plt.title(f"Image {i+1} | X: {starting_pos['x']}  Y: {starting_pos['y']}  Z: {best_z}")
    plt.axis('off')
    plt.show()

# Return to initial position
microscope.move(starting_pos)
assert microscope.position == starting_pos
print(f"Captured and saved 50 images in '{output_dir}'.")

```
