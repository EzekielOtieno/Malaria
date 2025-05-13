
import os
import time
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
from openflexure_microscope_client import MicroscopeClient
from ultralytics import YOLO

# Connect to the microscope
# microscope = MicroscopeClient("http://your-microscope-ip")  # Uncomment and update with your microscope IP

# Load the YOLO model
model = YOLO(r"C:\Users\Ezekiel\Desktop\model trained\MALARIA FINAL MODEL\best.pt")

# Set output directory
output_dir = Path.home() / "Desktop" / "T25"/"1"
os.makedirs(output_dir, exist_ok=True)
stats_file = output_dir / "statistics.txt"

# Parameters
step_x, step_y =600, 600
coarse_step = 100
fine_step = 50
coarse_steps = 2
fine_steps = 3
settle_time = 0.2
max_images = 10

num_x, num_y = 10, 10
scan_dir = 1
image_count = 0

# Stats
total_infected = total_non_infected = total_wbc = total_time = 0
statistics = []

# Movement helper
def move(x=None, y=None, z=None):
    pos = microscope.position
    if x is not None: pos['x'] = x
    if y is not None: pos['y'] = y
    if z is not None: pos['z'] = z
    microscope.move(pos)

def capture():
    return np.array(microscope.grab_image())

def laplacian_var(img):
    return cv2.Laplacian(img, cv2.CV_64F).var()

def evaluate_focus(z0, direction, step, num_steps):
    results = []
    for i in range(1, num_steps + 1):
        z = z0 + direction * step * i
        move(z=z)
        time.sleep(settle_time)
        img = capture()
        var = laplacian_var(img)
        results.append((var, img, z))
    return results

def find_peak_focus(z0, coarse_step, coarse_steps, fine_step, fine_steps):
    up = evaluate_focus(z0, 1, coarse_step, coarse_steps)
    down = evaluate_focus(z0, -1, coarse_step, coarse_steps)
    all_coarse = [(laplacian_var(capture()), capture(), z0)] + up + down

    best_var, best_img, best_z = max(all_coarse, key=lambda x: x[0])

    direction = 1 if best_z > z0 else -1
    fine = evaluate_focus(best_z, direction, fine_step, fine_steps)
    all_fine = [(best_var, best_img, best_z)] + fine

    best_var, best_img, best_z = max(all_fine, key=lambda x: x[0])

    move(z=best_z)

    print(f"🔍 Best focus variance: {best_var:.2f} at Z={best_z:.2f}")

    return best_img, best_z, best_var

def detect_and_save(img, count):
    img_pil = Image.fromarray(img)
    path = output_dir / f"image_{count}.png"
    img_pil.save(path)

    results = model(img, conf=0.25)
    pred_img = results[0].plot(labels=False)
    Image.fromarray(pred_img).convert("RGB").save(output_dir / f"image_{count}_predicted.png")

    pred = results[0]
    names = model.names
    infected = 0
    non_infected = 0
    wbc = 0

    if pred.boxes.cls is not None:
        class_ids = pred.boxes.cls.cpu().numpy().astype(int)
        for class_id in class_ids:
            class_name = names[class_id].lower().strip()
            if "infected" in class_name and "non" not in class_name:
                infected += 1
            elif "non imfected" in class_name or "non infected" in class_name:
                non_infected += 1
            elif "wbc" in class_name:
                wbc += 1

    print(f"🧫 Detected {infected} Infected RBCs, {non_infected} Non-Infected RBCs, {wbc} WBCs")

    return infected, non_infected, wbc

# Start scanning
start_pos = microscope.position
overall_start_time = time.time()

for j in range(num_y):
    for i in range(num_x):
        if image_count >= max_images:
            break

        t0 = time.time()

        dx = step_x * (i if scan_dir > 0 else (num_x - 1 - i))
        dy = step_y * j
        move(x=start_pos['x'] + dx, y=start_pos['y'] + dy)
        time.sleep(settle_time)

        initial_z = microscope.position['z']
        focused_img, best_z, best_var = find_peak_focus(initial_z, coarse_step, coarse_steps, fine_step, fine_steps)

        image_count += 1
        infected, non_infected, wbc = detect_and_save(focused_img, image_count)
        dt = round(time.time() - t0, 2)

        total_infected += infected
        total_non_infected += non_infected
        total_wbc += wbc
        total_time += dt

        stats = (
            f"Image {image_count}:
"
            f"  - Infected RBCs: {infected}
"
            f"  - Non-infected RBCs: {non_infected}
"
            f"  - WBCs: {wbc}
"
            f"  - Focus Variance: {best_var:.2f}
"
            f"  - Processing Time: {dt} seconds
"
            f"{'-' * 40}"
        )
        statistics.append(stats)
        print(stats)

    scan_dir *= -1

# Return to start
move(**start_pos)

# Save overall stats
overall_end_time = time.time()
overall_total_time = round(overall_end_time - overall_start_time, 2)
avg_time = round(total_time / image_count, 2)

with open(stats_file, "w") as f:
    f.write("Malaria Detection Statistics\n" + "=" * 40 + "\n")
    f.write("\n".join(statistics))
    f.write(
        f"\nTotal Statistics:\n"
        f"  - Total Infected RBCs: {total_infected}\n"
        f"  - Total Non-Infected RBCs: {total_non_infected}\n"
        f"  - Total WBCs: {total_wbc}\n"
        f"  - Total Time: {overall_total_time}s\n"
        f"  - Average Time per Image: {avg_time}s\n"
        f"{'=' * 40}"
    )

print(f"\n✅ Scan completed. Total Time: {overall_total_time}s. Images and statistics saved to: {output_dir}")
