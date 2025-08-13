import cv2
import numpy as np
import os
from ultralytics import YOLO
from sklearn.cluster import KMeans

# External method for detecting card dimensions using perspective
def detect_card_dimensions(image):
    CARD_WIDTH_MM  = 85.6
    CARD_HEIGHT_MM = 53.98

    orig = image.copy()
    gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edged = cv2.Canny(gray, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        raise RuntimeError("No contours found.")

    candidate_quads = []
    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            candidate_quads.append(approx)

    if not candidate_quads:
        raise RuntimeError("No quadrilateral contour found.")

    card_cnt = max(candidate_quads, key=cv2.contourArea)
    pts = card_cnt.reshape(4, 2)

    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    ordered = np.array([tl, tr, br, bl], dtype="float32")

    (tl, tr, br, bl) = ordered
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)

    pixel_width  = (widthA + widthB) / 2
    pixel_height = (heightA + heightB) / 2

    pixel_per_mm_w = pixel_width  / CARD_WIDTH_MM
    pixel_per_mm_h = pixel_height / CARD_HEIGHT_MM
    pixel_per_mm = (pixel_per_mm_w + pixel_per_mm_h) / 2

    card_width_mm  = pixel_width  / pixel_per_mm
    card_height_mm = pixel_height / pixel_per_mm

    return {
        'pixel_width': pixel_width,
        'pixel_height': pixel_height,
        'pixel_per_mm': pixel_per_mm,
        'bounding_box': cv2.boundingRect(card_cnt)
    }

class AdvanceCardShirtMeasurer:

    def __init__(self):
        self.card_width_mm = 85.6
        self.card_height_mm = 53.98
        self.load_yolo_model()

    def load_yolo_model(self):
        try:
            self.model = YOLO('model/yolov8n.pt')
            print("Successfully loaded YOLO model.")
        except Exception as e:
            print(f"YOLO model loading error: {e}")
            self.model = None

    def detect_person_yolo(self, image):
        if self.model is None:
            return None
        try:
            results = self.model(image)
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])
                        if class_id == 0 and confidence > 0.5:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            return {
                                'bbox': (int(x1), int(y1), int(x2 - x1), int(y2 - y1)),
                                'confidence': confidence
                            }
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return None

    def estimate_shirt_size(self, person_bbox, image):
        x, y, w, h = person_bbox
        shirt_start_ratio = 0.2
        shirt_end_ratio = 0.7
        shir_x = x + int(w * 0.1)
        shirt_w = int(w * 0.8)
        shirt_h = int(h * (shirt_end_ratio - shirt_start_ratio))
        shirt_y = y + int(h * shirt_start_ratio)
        shirt_x = max(0, shir_x)
        shirt_y = max(0, shirt_y)
        shirt_w = min(shirt_w, image.shape[1] - shirt_x)
        shirt_h = min(shirt_h, image.shape[0] - shirt_y)
        return (shirt_x, shirt_y, shirt_w, shirt_h)

    def refine_shirt_boundries(self, image, rough_shirt_area):
        x, y, w, h = rough_shirt_area
        shirt_area = image[y:y + h, x:x + w]
        if shirt_area.size == 0:
            return rough_shirt_area

        hsv_shirt_area = cv2.cvtColor(shirt_area, cv2.COLOR_BGR2HSV)
        pixels = hsv_shirt_area.reshape(-1, 3)

        try:
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[1]
            lower_bound = dominant_color - [10, 50, 50]
            upper_bound = dominant_color + [10, 50, 50]
            mask = cv2.inRange(hsv_shirt_area, lower_bound, upper_bound)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                refined_rect = cv2.boundingRect(largest_contour)
                refined_x = x + refined_rect[0]
                refined_y = y + refined_rect[1]
                refined_w = refined_rect[2]
                refined_h = refined_rect[3]
                return (refined_x, refined_y, refined_w, refined_h)
        except Exception as e:
            print(f"Error in refining shirt boundaries: {e}")
        return rough_shirt_area

    def process_image(self, image_path, show_results=True):
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Image not found or could not be read.")
            return {"error": "Image not found or could not be read."}

        height, width = image.shape[:2]
        max_size = 800
        if max(height, width) > max_size:
            scale = max_size / float(max(height, width))
            image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)

        try:
            card_info = detect_card_dimensions(image)
        except Exception as e:
            print(f"No card detected: {e}")
            return {"error": "No card detected."}

        card_rect = card_info['bounding_box']
        card_width_pixels = card_info['pixel_width']
        pixel_to_mm_ratio = card_info['pixel_per_mm']

        person_info = self.detect_person_yolo(image)
        if person_info is None:
            print("No person detected.")
            return {"error": "No person detected."}

        rough_shirt_area = self.estimate_shirt_size(person_info['bbox'], image)
        shirt_area = self.refine_shirt_boundries(image, rough_shirt_area)
        shirt_width_mm = shirt_area[2] / pixel_to_mm_ratio
        shirt_height_mm = shirt_area[3] / pixel_to_mm_ratio

        if show_results:
            result_image = image.copy()
            cv2.rectangle(result_image, (card_rect[0], card_rect[1]),
                          (card_rect[0] + card_rect[2], card_rect[1] + card_rect[3]), (0, 255, 0), 2)
            cv2.rectangle(result_image, (shirt_area[0], shirt_area[1]),
                          (shirt_area[0] + shirt_area[2], shirt_area[1] + shirt_area[3]), (255, 0, 0), 2)
            cv2.putText(result_image, f"Card: {self.card_width_mm:.2f}mm x {self.card_height_mm:.2f}mm",
                        (card_rect[0], card_rect[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            bbox = person_info['bbox']
            cv2.rectangle(result_image, (bbox[0], bbox[1]),
                          (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 0, 255), 2)
            cv2.putText(result_image, f"Shirt: {shirt_width_mm:.2f}mm x {shirt_height_mm:.2f}mm",
                        (shirt_area[0], shirt_area[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            cv2.imshow("Detected Card and Shirt", result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return {
            'sucsees': True,
            'card_dimensions_mm': {
                'width_mm': self.card_width_mm,
                'height_mm': self.card_height_mm
            },
            'shirt_dimensions_mm': {
                'width_mm': shirt_width_mm,
                'height_mm': shirt_height_mm
            },
            'shirt_to_card_ratio': {
                'width_ratio': round(shirt_width_mm / self.card_width_mm, 2),
                'height_ratio': round(shirt_height_mm / self.card_height_mm, 2)
            },
            'pixel_to_mm_ratio': round(pixel_to_mm_ratio, 4),
            'person_confidence': round(person_info['confidence'], 2)
        }

if __name__ == "__main__":
    measurer = AdvanceCardShirtMeasurer()
    # read images from folder  
    # image_folder = "test_images" 
    # image_files = [
    #     os.path.join(image_folder, img)
    #     for img in os.listdir(image_folder)
    #     if img.lower().endswith(('.png', '.jpg', '.jpeg'))
    #     ]
    # for img_path in image_files:
    #     results = measurer.process_image(img_path)  # اینجا مسیر تکی میدیم
    #     print(results)
    image_path  = "test_images/3.jpg"  # Change this to your image path
    results = measurer.process_image(image_path)
    if "error" in results:
        print(f"Error: {results['error']}")
    else:
        print(f"Card Dimensions: {results['card_dimensions_mm']}")
        print(f"Shirt Dimensions: {results['shirt_dimensions_mm']}")
        print(f"Shirt to Card Ratio: {results['shirt_to_card_ratio']}")
        print(f"Pixel to mm Ratio: {results['pixel_to_mm_ratio']}")
        print(f"Person Confidence: {results['person_confidence']}")
