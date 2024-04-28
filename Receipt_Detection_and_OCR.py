import re
from typing import Any, List

import cv2
import numpy as np
import pytesseract
from cv2.typing import MatLike
from numpy.typing import NDArray
from pytesseract import Output
from skimage.filters import threshold_local
from prettytable import PrettyTable
import matplotlib.pyplot as plt
from ultralytics import YOLO



pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'



model = YOLO("D:/Yolov8/runs/detect/train6/weights/best.pt")

image_path = "C:/Users/Ramez/Downloads/large-receipt-image-dataset-SRD/z1141-receipt.jpg"

image = cv2.imread(image_path)


results = model.predict(image_path)
for result in results:
    boxes = result.boxes  # Bounding box outputs
    for box in boxes:
        bbox_coordinates = box.xyxy[0]
        print(bbox_coordinates)
  


    masks = result.masks  # Segmentation mask outputs
    keypoints = result.keypoints  # Pose keypoints
    probs = result.probs  # Classification probabilities
    result.show() 



x1, y1, x2, y2 = bbox_coordinates

# Convert the coordinates to integers
x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

# Crop the image
cropped_image = image[y1:y2, x1:x2]


# # Display the cropped image
# cv2.imshow("Cropped Receipt", cropped_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('D:/check_ocr/cropped_image.bmp', cropped_image)


def enhance_txt(img):
    w = img.shape[1]
    h = img.shape[0]
    w1 = int(w*0.05)
    w2 = int(w*0.95)
    h1 = int(h*0.05)
    h2 = int(h*0.95)
    ROI = img[h1:h2, w1:w2]  
    threshold = np.mean(ROI) * 0.98

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (1, 1), 0)
    edged = 255 - cv2.Canny(blurred, 100, 150, apertureSize=7)

    thresh, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)

    return binary

def enhance_txt_2(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    denoised = cv2.fastNlMeansDenoising(binary, None, h=10, templateWindowSize=7, searchWindowSize=21)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel, iterations=2)

    edges = cv2.Canny(closed, 30, 150)

    enhanced = cv2.bitwise_or(denoised, edges)

    return enhanced


image_path = (

'D:/check_ocr/cropped_image.bmp'

)



image = cv2.imread(image_path)
enhanced = enhance_txt(image)
enhanced_2 = enhance_txt_2(image)


def opencv_resize(image: MatLike, ratio: float):
    height = int(image.shape[0] * ratio)
    width = int(image.shape[1] * ratio)
    dim = (width, height)
    return cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


def approximate_contour(contour: MatLike):
    peri = cv2.arcLength(contour, True)
    return cv2.approxPolyDP(contour, 0.032 * peri, True)


def get_receipt_contour(contours: List[MatLike]):
    for c in contours:
        approx = approximate_contour(c)
        if len(approx) == 4:
          return approx
        else:
            points = approximate_contour(contours[0])

            points = points.reshape(-1, 2)

            sorted_points = points[np.argsort(points[:, 0]), :]

            tl = sorted_points[np.argmin(sorted_points.sum(axis=1))]
            br = sorted_points[np.argmax(sorted_points.sum(axis=1))]

            tr = sorted_points[np.argmin(np.diff(sorted_points, axis=1))]
            bl = sorted_points[np.argmax(np.diff(sorted_points, axis=1))]

            rectangle = np.array([tl, tr, br, bl], dtype="float32")
            return rectangle


def contour_to_rect(contour: MatLike | NDArray[Any], resize_ratio: float):
    pts = contour.reshape(4, 2)
    rect = np.zeros((4, 2), dtype = "float32")
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect / resize_ratio


def wrap_perspective(img: MatLike, rect: NDArray[np.floating[Any]]):
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
   
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
      [0, 0],
      [maxWidth - 1, 0],
      [maxWidth - 1, maxHeight - 1],
      [0, maxHeight - 1]], dtype = "float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(img, M, (maxWidth, maxHeight))


def bw_scanner(image: MatLike):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    T = threshold_local(gray, 21, offset = 5, method = 'gaussian')
    return (gray > T).astype("uint8") * 255


def convert_to_float(value: str):
    amounts = re.findall(r'\d+\s?[\.,°]\s?\d{2}\b', value)
    floats = [float(amount.replace('°', '.').replace(' ', '').replace('° ','.').replace('..','.').replace(',','.')) for amount in amounts]
    return floats


def check_ocr(file_name: str):
    image = cv2.imread(file_name)

    if image is None:
        print("Could not open or find the image")
        exit(0)
    
    resize_ratio = 500 / image.shape[0]
    original = image.copy()
    image = opencv_resize(image, resize_ratio)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9,9))
    dilated = cv2.dilate(blurred, rectKernel)
    edged = cv2.Canny(dilated, 50, 150, apertureSize = 3)
    
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    largest_contours = sorted(contours, key = cv2.contourArea, reverse = True)[:3]
    
    if cv2.contourArea(largest_contours[0]) < 50000:
        image_height, image_width = image.shape[:2]
        entire_contour = np.array([[0, 0], [0, image_height], [image_width, image_height], [image_width, 0]], dtype=np.int32)
        largest_contours[0] = entire_contour
    
    receipt_contour = get_receipt_contour(largest_contours)
    
    if receipt_contour is not None and len(receipt_contour) > 0:
        image_with_receipt_contour = image.copy()
        cv2.drawContours(image_with_receipt_contour, [receipt_contour.astype(int)], -1, (0, 255, 0), 3)
    else:
        print("No valid receipt contour found.")

    scanned = wrap_perspective(original.copy(), contour_to_rect(receipt_contour, resize_ratio))

    result = bw_scanner(scanned)

    
    d = pytesseract.image_to_data(enhanced.copy(), output_type = Output.DICT,config = r'--psm 11')
    
    n_boxes = len(d['level'])
    boxes = cv2.cvtColor(result.copy(), cv2.COLOR_BGR2RGB)
    for i in range(n_boxes):
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        boxes = cv2.rectangle(boxes, (x,y), (x + w, y + h), (0,255,0), 2)



    extracted_text = pytesseract.image_to_string(result.copy(), config=r'--oem 3 --psm 6')

    extracted_text_2 = pytesseract.image_to_string(enhanced.copy(), config=r'--oem 3 --psm 6')

    extracted_text_3 = pytesseract.image_to_string(enhanced_2.copy(), config=r'--oem 3 --psm 6')


    def extract_items_and_costs(extracted_text):
        item_cost_pairs = []
        potential_item_name = None  
        potential_item_taken = False
        exclusion_list = ["bank", "total", "promo", "vat", "change", "recyclable","subtotal","tax","tips","amount","visa","service","gratuity","balance","kg","cash","totl","void","credit","iotal","subtotam","totaal","payment","surcharge","subtota","tip"]

        for line in extracted_text.splitlines():
            


            if any(exclude in line.lower() for exclude in exclusion_list):
                potential_item_name = None
                continue  
            if re.search(r"\d+\s?[.,°]\s?\d{2}(?:[\$§¥])?", line):
                line = re.sub(r"(\d+)[.,°']\s*(\d{2})", r'\1.\2', line)
                line = re.sub(r'(\d+)\s+(\d{2})', r'\1.\2', line)

                

                cost = re.findall(r"(?:-?[\$§-“])?\d+\.\d{2}(?:[\$§¥])?\b", line)
                cost = [float(re.sub(r'[^\d.]', '', c)) for c in cost]
                cost = [f"{c:.2f}" for c in cost]
               
                item_name = re.sub(r"^\d+\s?[.,°]\s?\d{2}(?:[\$§¥])?\s*|EUR", "", line).strip()
                item_name = re.sub(r"[^a-zA-Z\s]", "", item_name)

                zzz = item_name.replace(" ","")              

                
                if cost :
                    if len(zzz) >= 4 and item_name.lower() not in [exclusion.lower() for exclusion in exclusion_list]:

                        item_cost_pairs.append((item_name, cost[-1]))
                        potential_item_name = None

                    elif potential_item_name :
                            
                            item_cost_pairs.append((potential_item_name, cost[-1]))
                            potential_item_name = None  
             
            else:
                potential_item_name = line.strip()


        
        return item_cost_pairs

    


    

    item_cost_pairs = extract_items_and_costs(extracted_text)
    for item, cost in item_cost_pairs:
         f"Item: {item}, Cost: {cost}"

    item_cost_pairs_2 = extract_items_and_costs(extracted_text_2)
    for item, cost in item_cost_pairs_2:
        f"Item: {item}, Cost: {cost}"    
    
    item_cost_pairs_3 = extract_items_and_costs(extracted_text_3)
    for item, cost in item_cost_pairs_3:

        f"Item: {item}, Cost: {cost}"

  
    t = PrettyTable(['Item', 'Cost'])

    max_length_list = max(item_cost_pairs, item_cost_pairs_2, item_cost_pairs_3, key=len)

    for item, cost in max_length_list:
        t.add_row([item, cost])

    print(t)


   
    def extract_tax_vat(extracted_text):
        tax_vat_pairs = []
        tax_vat_list = ["tax", "vat","tak"]

        for line in extracted_text.splitlines():
            if re.search(r"[0-9]*\.[0-9]|[0-9]*\,[0-9]", line):
                line = re.sub(r'(\d+)[.,°]\s*(\d{2})', r'\1.\2', line)
                cost = re.findall(r'\d+\.\d{2}\b', line)
                item_name = re.sub(r'\d+\s?[\.,°]\s?\d{2}\b.*$', '', line).strip()
                item_name = re.sub(r'[^a-zA-Z\s]', '', item_name)
                
                if cost and any(tax_or_vat in item_name.lower() for tax_or_vat in tax_vat_list):
                    tax_vat_pairs.append((item_name, cost[-1]))
        
        return tax_vat_pairs



    tax_vat_pairs = extract_tax_vat(extracted_text)
    tax_vat_pairs_2 = extract_tax_vat(extracted_text_2)
    tax_vat_pairs_3 = extract_tax_vat(extracted_text_3)

    
    t = PrettyTable(['Item', 'Amount'])
    t.title = "Tax and VAT Details"
    max_length_list = max(tax_vat_pairs, tax_vat_pairs_2, tax_vat_pairs_3, key=len)
    for item, cost in max_length_list:
        t.add_row([item, cost])
    print(t)  




prices = check_ocr('D:/check_ocr/cropped_image.bmp')

