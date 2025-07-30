import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar

filename = input("Enter the image filename: ").strip()
img = cv2.imread(filename)
if img is None:
    print(f"File '{filename}' not found.")
    exit()


decoded_objects = pyzbar.decode(img)
for obj in decoded_objects:

    if obj.type == "CODE39":

        points = obj.polygon
        if len(points) > 4:
            hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
            points = hull
        points = np.array(points, dtype=np.int32)
        cv2.polylines(img, [points], isClosed=True, color=(0, 255, 0), thickness=2)

        barcode_data = obj.data.decode("utf-8")
        barcode_type = obj.type
        text = f"{barcode_data} ({barcode_type})"
        x, y, w, h = obj.rect
        cv2.putText(img, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        print(f"Detected Code 39 Barcode: {barcode_data}")

cv2.imshow("Barcode Scanner", img)
cv2.waitKey(0)
cv2.destroyAllWindows()