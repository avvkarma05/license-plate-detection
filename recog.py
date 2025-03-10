import cv2
import numpy as np
import pytesseract

def find_license_plate(image):
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    edges = cv2.Canny(blur, 50, 150)
    
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    
    license_plate = None
    
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        if len(approx) == 4:  # Rectangle has 4 corners
            license_plate = approx
            break
    
    return license_plate

def read_license_plate(image, plate_contour):
    if plate_contour is None:
        return "No license plate found"
    
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [plate_contour], 0, 255, -1)
    
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    
    x, y, w, h = cv2.boundingRect(plate_contour)
    plate = masked_image[y:y+h, x:x+w]
    
    text = pytesseract.image_to_string(plate, config='--psm 8')
    
    return text.strip()

def recognize_license_plate(image_path):

    image = cv2.imread(image_path)
    
    plate_contour = find_license_plate(image)
    
    plate_text = read_license_plate(image, plate_contour)
    
    if plate_contour is not None:
        cv2.drawContours(image, [plate_contour], 0, (0, 255, 0), 2)
        cv2.putText(image, plate_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("License Plate", image)
    cv2.waitKey(0)
    
    return plate_text

# Example usage - this is where you'd run your program
if __name__ == "__main__":
    # Change this to your test image path
    image_path = "D:\\AV\\test_img\\001.jpg"  # This is an example image path
    result = recognize_license_plate(image_path)
    print(f"License plate text: {result}")