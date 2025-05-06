import cv2
import easyocr
import numpy as np
import re
import os
import torch
import sys
import time
from pathlib import Path
from flask import Flask, request, jsonify
import base64

# Try importing YOLO, but don't fail if not available
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("YOLO module loaded successfully")
except ImportError:
    YOLO_AVAILABLE = False
    print("YOLO module not available. Falling back to traditional detection methods.")

class TunisianPlateDetector:
    def __init__(self):
        try:
            # Initialize CUDA if available
            if not torch.cuda.is_available():
                print("WARNING: CUDA is not available. Using CPU instead.")
                self.device = 'cpu'
            else:
                try:
                    torch.cuda.init()
                    self.device = 'cuda'
                    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                except Exception as e:
                    print(f"CUDA initialization failed: {e}")
                    self.device = 'cpu'
            
            # Memory optimization: Only initialize components when needed
            self.reader = None
            self.plate_detector = None
            self.plate_cascade = None
            
            # Initialize cascade classifier as it's lightweight
            self._init_cascade()
            
            # Tunisia letters and patterns
            self.tunisia_letters = ['ت', 'و', 'ن', 'س']
            self.tunisia_pattern = re.compile(r'[تونس]+')
            self.number_pattern = re.compile(r'\d+')
            
            # Parameters
            self.min_plate_area = 500
            self.min_confidence = 0.2
            
        except Exception as e:
            print(f"Error during initialization: {str(e)}")
            sys.exit(1)

    def _init_cascade(self):
        try:
            cascade_path = os.path.join(os.path.dirname(__file__), "model/haarcascade_russian_plate_number.xml")
            if not os.path.exists(cascade_path):
                cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
                
            if os.path.exists(cascade_path):
                self.plate_cascade = cv2.CascadeClassifier(cascade_path)
                print(f"Cascade classifier loaded from: {cascade_path}")
            else:
                self.plate_cascade = cv2.CascadeClassifier()
        except Exception as e:
            print(f"Error loading cascade: {e}")
            self.plate_cascade = None
    
    def _init_easyocr(self):
        if self.reader is None:
            try:
                print("Initializing EasyOCR...")
                gpu_status = True if self.device == 'cuda' else False
                self.reader = easyocr.Reader(['ar', 'en'], gpu=gpu_status, 
                                           recog_network='arabic_g1')
                print("EasyOCR initialized successfully")
            except Exception as e:
                print(f"EasyOCR initialization failed: {e}")
                self.reader = None
                
    def _init_yolo(self):
        if not YOLO_AVAILABLE:
            print("YOLO not available")
            return
            
        if self.plate_detector is None:
            try:
                print("Initializing YOLO detector...")
                model_path = os.path.join(os.path.dirname(__file__), "model/license_plate_yolov8n.pt")
                if os.path.exists(model_path):
                    self.plate_detector = YOLO(model_path)
                    print(f"YOLO model loaded from: {model_path}")
                else:
                    print(f"YOLO model not found at {model_path}. Using general object detection.")
                    self.plate_detector = YOLO('yolov8n.pt')
                print("YOLO initialized successfully")
            except Exception as e:
                print(f"YOLO initialization failed: {e}")
                self.plate_detector = None

    def ensure_image_format(self, img):
        """Make sure image is in the correct format for processing"""
        try:
            if img is None or img.size == 0:
                print("Invalid image input")
                return None
            
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)
                
            return img
        except Exception as e:
            print(f"Error in image format conversion: {e}")
            return np.ones((100, 100, 3), dtype=np.uint8) * 255

    def detect_plate(self, img):
        """Detect license plates using available methods"""
        try:
            height, width = img.shape[:2]
            max_dimension = 1600
            if max(height, width) > max_dimension:
                scale = max_dimension / max(height, width)
                img = cv2.resize(img, None, fx=scale, fy=scale)
            
            detected_plates = []
            
            # Try YOLO detection if available
            if YOLO_AVAILABLE and self.plate_detector is not None:
                try:
                    detected_plates = self.detect_with_yolo(img)
                except Exception as e:
                    print(f"YOLO detection failed: {e}")
            
            # If no plates found or YOLO not available, try cascade
            if not detected_plates:
                try:
                    detected_plates = self.detect_with_cascade(img)
                except Exception as e:
                    print(f"Cascade detection failed: {e}")
            
            # If still no plates, try direct OCR on the full image
            if not detected_plates:
                try:
                    print("Attempting direct OCR on full image...")
                    h, w = img.shape[:2]
                    detected_plates.append((0, 0, w, h, 0.3))
                except Exception as e:
                    print(f"Direct OCR preparation failed: {e}")
            
            return detected_plates
            
        except Exception as e:
            print(f"Error in plate detection: {str(e)}")
            return []
    
    def detect_with_yolo(self, img):
        """Detect license plates using YOLO"""
        self._init_yolo()
        if not YOLO_AVAILABLE or self.plate_detector is None:
            return []
        
        try:
            results = self.plate_detector(img, conf=self.min_confidence)
            yolo_plates = []
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if box.conf.item() > self.min_confidence:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        w, h = x2 - x1, y2 - y1
                        aspect_ratio = w / float(h) if h > 0 else 0
                        
                        if 1.5 <= aspect_ratio <= 6.0 and w * h >= self.min_plate_area:
                            yolo_plates.append((x1, y1, w, h, box.conf.item()))
                                    
            return yolo_plates
        except Exception as e:
            print(f"Error in YOLO detection: {e}")
            return []
    
    def detect_with_cascade(self, img):
        """Detect license plates using Haar cascade"""
        if self.plate_cascade is None:
            return []
        
        try:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.equalizeHist(img_gray)
            img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
            
            all_detections = []
            
            cascade_plates = self.plate_cascade.detectMultiScale(
                img_gray, 
                scaleFactor=1.1,
                minNeighbors=3,
                minSize=(60, 20),
                maxSize=(300, 100)
            )
            all_detections.extend(cascade_plates)
            
            filtered_plates = []
            seen_regions = set()
            
            for (x, y, w, h) in all_detections:
                region_id = f"{x//10}_{y//10}_{w//10}_{h//10}"
                
                if region_id in seen_regions:
                    continue
                
                seen_regions.add(region_id)
                area = w * h
                aspect_ratio = w / float(h)
                
                if 1.5 <= aspect_ratio <= 6.0 and area > self.min_plate_area:
                    filtered_plates.append((x, y, w, h, 0.5))
                    
            return filtered_plates
        except Exception as e:
            print(f"Error in cascade detection: {e}")
            return []
    
    def preprocess_image(self, img):
        """Apply minimal preprocessing techniques for detection"""
        try:
            img = self.ensure_image_format(img)
            if img is None:
                return [np.ones((100, 100, 3), dtype=np.uint8) * 255]
            
            preprocessed_images = []
            preprocessed_images.append(img.copy())
            
            # Add grayscale variant
            try:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                preprocessed_images.append(gray_3ch)
            except Exception as e:
                print(f"Grayscale conversion error: {e}")
            
            return preprocessed_images
        except Exception as e:
            print(f"Critical error in preprocessing: {str(e)}")
            if img is not None and img.size > 0:
                return [img.copy()]
            return [np.ones((100, 100, 3), dtype=np.uint8) * 255]
    
    def process_ocr_results(self, ocr_results):
        """Process OCR results to extract plate information with confidence scoring"""
        # Sort results by confidence
        sorted_results = sorted(ocr_results, key=lambda x: x[2], reverse=True)
        
        # Debug OCR results
        print("OCR Results:", [(text, conf) for _, text, conf in sorted_results])
        
        # Look for تونس or Arabic letters with confidence scores
        tunisia_text = ""
        confidence = 0.0
        
        # First check for exact تونس match
        for _, text, prob in sorted_results:
            if 'تونس' in text:
                tunisia_text = "تونس"
                confidence = prob
                print(f"Found exact 'تونس' match in: {text}")
                break
        
        # If no exact match, look for letters or variants
        if not tunisia_text:
            for _, text, prob in sorted_results:
                cleaned_text, replaced, match_conf = self.detect_tunisia_text(text)
                if replaced:
                    tunisia_text = cleaned_text
                    confidence = match_conf * prob
                    print(f"Found Tunisia variant in: {text}, replaced with: {cleaned_text}")
                    break
                    
        # If still no match, force include تونس in the result
        if not tunisia_text:
            tunisia_text = "تونس"
            confidence = 0.5
            print("Forcing inclusion of 'تونس' as no match was found")
        
        # Extract all numbers using regex
        all_text = ' '.join([item[1] for item in sorted_results])
        numbers = re.findall(r'\d+', all_text)
        
        # Remove duplicates while preserving order
        unique_numbers = []
        seen = set()
        for num in numbers:
            if num not in seen:
                unique_numbers.append(num)
                seen.add(num)
        numbers = unique_numbers
        
        # Sort numbers by value (not length) and handle duplicates
        # If we have the same number twice, keep only one instance
        try:
            numbers = sorted([int(num) for num in numbers])
            numbers = [str(num) for num in numbers]
        except ValueError:
            # If conversion fails, keep original order
            pass
        
        # Format the complete plate text - always include تونس
        if numbers:
            if len(numbers) >= 2:
                # Take the smallest and largest numbers for the standard format
                smallest_num = min(numbers, key=lambda x: int(x) if x.isdigit() else 9999)
                largest_num = max(numbers, key=lambda x: int(x) if x.isdigit() else 0)
                
                # Standard Tunisian format: small_num تونس large_num
                formatted_text = f"{smallest_num} {tunisia_text} {largest_num}"
                confidence_score = 0.7  # Higher confidence as this matches expected format
            elif len(numbers) == 1:
                # If only one number is found, follow standard format with default values
                formatted_text = f"{numbers[0]} {tunisia_text}"
                confidence_score = 0.5
            else:
                formatted_text = f"{تونس}"
                confidence_score = 0.3
        else:
            formatted_text = tunisia_text
            confidence_score = 0.2
            
        print(f"Final plate text: '{formatted_text}' with confidence: {confidence_score}")
        
        return formatted_text, numbers, tunisia_text, confidence_score
    
    def detect_tunisia_text(self, text):
        """Detect if text contains تونس or its letters and replace if needed"""
        # Pre-normalize the text to handle common OCR errors
        normalized_text = text.replace(' ', '')
        
        # Direct match for تونس (most confident)
        if 'تونس' in normalized_text:
            return "تونس", True, 1.0
        
        # Check for variants with small spelling errors - add more variants
        variants = ['ٹونس', 'توسن', 'نوست', 'ونست', 'تؤنس', 'ٺونس', 'تويس', 'نوسن', 'يوس', 'نوس']
        for variant in variants:
            if variant in normalized_text:
                print(f"Found variant '{variant}' of تونس")
                return "تونس", True, 0.9
        
        # Check for individual letters from تونس
        matches = []
        for letter in self.tunisia_letters:
            if letter in normalized_text:
                matches.append(letter)
        
        # More aggressive matching - even with just 1-2 letters
        if matches:
            match_ratio = len(matches) / len(self.tunisia_letters)
            confidence = min(0.8, 0.3 + match_ratio * 0.5)  # More lenient threshold
            print(f"Found letters {', '.join(matches)} from تونس in '{text}'")
            return "تونس", True, confidence
        
        # Any Arabic text could be a corrupted version of تونس
        arabic_pattern = re.compile(r'[\u0600-\u06FF]+')
        arabic_match = arabic_pattern.search(normalized_text)
        if arabic_match:
            arabic_text = arabic_match.group(0)
            if len(arabic_text) >= 1:  # Even with just one Arabic character
                print(f"Found Arabic text: {arabic_text}, assuming it's related to تونس")
                return "تونس", True, 0.4
        
        return text, False, 0.0
    
    def process_image_array(self, img):
        """Process an image array directly with improved detection and OCR"""
        if img is None:
            return None, 0.0
            
        try:
            # First ensure valid image format
            img = self.ensure_image_format(img)
            if img is None:
                return None, 0.0
            
            # Detect plate regions with confidence scores
            plate_regions = self.detect_plate(img)
            if not plate_regions:
                print("No license plates detected")
                return None, 0.0
                
            # Initialize OCR only when needed
            self._init_easyocr()
            if self.reader is None:
                print("Failed to initialize OCR")
                return None, 0.0
                
            best_confidence = 0.0
            best_plate_text = None
            
            for idx, (x, y, w, h, detection_conf) in enumerate(plate_regions):
                try:
                    # Make sure coordinates are valid
                    x, y, w, h = int(x), int(y), int(w), int(h)
                    if x < 0: x = 0
                    if y < 0: y = 0
                    if w <= 0 or h <= 0 or x+w > img.shape[1] or y+h > img.shape[0]:
                        print(f"Invalid plate region #{idx}: ({x}, {y}, {w}, {h}) for image of size {img.shape}")
                        continue
                    
                    # Extract plate region
                    plate_roi = img[y:y+h, x:x+w].copy()
                    
                    # Apply preprocessing to get multiple variants
                    preprocessed_variants = self.preprocess_image(plate_roi)
                    
                    # Apply OCR
                    ocr_results = []
                    for variant in preprocessed_variants:
                        try:
                            results = self.reader.readtext(variant)
                            ocr_results.extend(results)
                        except Exception as e:
                            print(f"OCR error: {e}")
                    
                    if ocr_results:
                        # Process the text with confidence scoring
                        plate_text, numbers, tunisia_text, text_conf = self.process_ocr_results(ocr_results)
                        
                        # Combine detection and OCR confidence
                        combined_conf = detection_conf * 0.3 + text_conf * 0.7
                        
                        if combined_conf > best_confidence:
                            best_confidence = combined_conf
                            best_plate_text = plate_text
                except Exception as e:
                    print(f"Error processing plate region #{idx}: {e}")
                    continue
                        
            if best_plate_text:
                return best_plate_text, best_confidence
            
            return None, 0.0
        except Exception as e:
            print(f"Error in image processing: {e}")
            return None, 0.0

# Initialize Flask application with memory-optimized settings
app = Flask(__name__)

# Create detector instance only when needed (lazy loading)
detector = None

def get_detector():
    global detector
    if detector is None:
        detector = TunisianPlateDetector()
    return detector

@app.route('/detect_plate', methods=['POST'])
def detect_plate():
    try:
        # Initialize detector when needed
        detector = get_detector()
        
        # Check if the request contains an image
        if 'image' not in request.files and 'image' not in request.json:
            return jsonify({'error': 'No image provided'}), 400
        
        # Handle file upload
        if 'image' in request.files:
            file = request.files['image']
            img_bytes = file.read()
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # Handle base64 encoded image
        elif 'image' in request.json:
            base64_img = request.json['image']
            if base64_img.startswith('data:image'):
                base64_img = base64_img.split(',')[1]
            img_bytes = base64.b64decode(base64_img)
            nparr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Process the image
        plate_text, confidence = detector.process_image_array(img)
        
        if plate_text is None:
            return jsonify({
                'success': False,
                'plateText': None,
                'confidence': 0,
                'noPlateDetected': True
            })
        
        # Make sure to explicitly encode تونس in the response
        print(f"Sending response with plate text: {plate_text}")
        return jsonify({
            'success': True,
            'plateText': plate_text,
            'confidence': float(confidence),
            'noPlateDetected': False
        })
    
    except Exception as e:
        print(f"Error processing request: {e}")
        return jsonify({'error': str(e)}), 500

# Add a basic root route
@app.route('/', methods=['GET'])
def index():
    return jsonify({
        'service': 'Tunisian License Plate Detection API',
        'status': 'active',
        'sample': '123 تونس 4567',  # Test Arabic encoding
        'memory_optimized': True
    })

if __name__ == '__main__':
    import gc
    # Force garbage collection to free memory
    gc.collect()
    
    # Get port from environment variable (for Render.com compatibility)
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
else:
    # This will allow importing the app from other files
    # The app object is already initialized above
    pass

