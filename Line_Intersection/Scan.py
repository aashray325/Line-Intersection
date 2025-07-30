import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

filename = input("Enter the image filename: ").strip()

if not os.path.isfile(filename):
    print(f"File '{filename}' not found.")
    exit()

def line_intersect(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2
    if (x2 - x1) == 0 or (x4 - x3) == 0:
        return None

    m1 = (y2 - y1) / (x2 - x1)
    m2 = (y4 - y3) / (x4 - x3)
    if m1 == m2: 
        return None

    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / ((y4 - y3) * (x1 - x2) - (y2 - y1) * (x3 - x4))
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / ((y4 - y3) * (x1 - x2) - (y2 - y1) * (x3 - x4))


def is_axis_line(line, angle_thresh=10):
    x1, y1, x2, y2 = line
    dx = x2 - x1
    dy = y2 - y1
    angle = np.degrees(np.arctan2(dy, dx))
    if abs(angle) < angle_thresh or abs(abs(angle) - 180) < angle_thresh:
        return True 
    if abs(abs(angle) - 90) < angle_thresh:
        return True  
    return False

def intersection_point(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return None
    px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4)) / denom
    tol = 1
    if (min(x1, x2)-tol <= px <= max(x1, x2)+tol and
        min(y1, y2)-tol <= py <= max(y1, y2)+tol and
        min(x3, x4)-tol <= px <= max(x3, x4)+tol and
        min(y3, y4)-tol <= py <= max(y3, y4)+tol):
        return (int(round(px)), int(round(py)))
    return None

img = cv2.imread(filename)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh_inv = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

lines = cv2.HoughLinesP(thresh_inv, 1, np.pi / 180, threshold=50, minLineLength=30, maxLineGap=20)

height, width = img.shape[:2]

points = []
adjusted_lines = []
if lines is not None:
    lines = lines[:, 0]
    for line in lines:
        if not is_axis_line(line):
            x1, y1, x2, y2 = line
            adjusted_lines.append([x1, y1, x2, y2])
    for i in range(len(adjusted_lines)):
        for j in range(i + 1, len(adjusted_lines)):
            pt = intersection_point(adjusted_lines[i], adjusted_lines[j])
            if pt:
                points.append(pt)

unique_points = set(points)
num_intersections = len(unique_points)
print(f"Number of intersections: {num_intersections}")

plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.title(f"Intersections Highlighted (Total: {num_intersections})")
plt.axis('on')
for point in unique_points:
    plt.scatter(point[0], point[1], c='red', s=100, marker='o', linewidths=2)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
