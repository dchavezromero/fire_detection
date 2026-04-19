"""
This is a script to connect the UAV coordinates outputted for input into UBC Cascade CNN.
"""

import requests
import cv2
import numpy as np
import math


def fetch_esri_satellite_image(lat, lon, buffer_meters=150, img_size=800):
    """
    High-quality square satellite image from the Esri API

    Parameters:
    - lat, lon: GPS coordinates from the drone.
    - buffer_meters: The distance from the center point to the edge of the bounding box.
    - img_size: The width and height of the returned square image in pixels.
    """

    # Esri World Imagery Export API endpoint
    url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"

    # Approximate conversion: 1 degree latitude = ~111,320 meters
    lat_degree_dist = 111320.0
    lon_degree_dist = 111320.0 * math.cos(math.radians(lat))

    # Calculate bounding box (bbox) offsets
    lat_offset = buffer_meters / lat_degree_dist
    lon_offset = buffer_meters / lon_degree_dist

    # Bounding box: xmin, ymin, xmax, ymax. longitude is X and latitude is Y.
    xmin = lon - lon_offset
    ymin = lat - lat_offset
    xmax = lon + lon_offset
    ymax = lat + lat_offset
    bbox = f"{xmin},{ymin},{xmax},{ymax}"

    # API Parameters
    params = {
        "bbox": bbox,
        "bboxSR": "4326",  # Bounding box in standard GPS format
        "imageSR": "102100",  # Force image output to Web Mercator (Esri's native map format)
        "size": f"{img_size},{img_size}",
        "format": "png",
        "f": "image"
    }

    # Bypass standard bot-filters that sometimes throw 500/403 errors
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    print(f"[INFO] Requesting Esri image for coordinates: {lat}, {lon}")

    # Catching request with headers included
    response = requests.get(url, params=params, headers=headers)

    if response.status_code == 200:
        # Convert raw network bytes directly into OpenCV-compatible NumPy array
        image_array = np.asarray(bytearray(response.content), dtype="uint8")
        cv_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        print("[INFO] Satellite image successfully retrieved.")
        return cv_image
    else:
        # Adding due to failure potential
        print(f"[ERROR] Esri API request failed with status code: {response.status_code}")
        print(f"[DEBUG] Server response: {response.text}")
        return None


# Quick Demo
if __name__ == "__main__":
    # Put in the Empire State Building
    test_lat = 40.7484
    test_lon = -73.9857

    satellite_img = fetch_esri_satellite_image(test_lat, test_lon, buffer_meters=150, img_size=800)

    if satellite_img is not None:
        cv2.imshow("Esri World Imagery", satellite_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()