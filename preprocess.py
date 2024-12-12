import cv2
import os

def apply_clahe(image_folder, output_folder):
    """
    Applies CLAHE to all images in a folder and saves the results to a new folder.

    Args:
      image_folder: Path to the folder containing the images.
      output_folder: Path to the folder where the processed images will be saved.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(image_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Create a CLAHE object 
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl1 = clahe.apply(img)

            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, cl1)

image_folder = "D:/Northeastern/Semester_1/DEM/Final_Project/coco_minitrain_25k/images/train2017" 
output_folder = 'D:/Northeastern/Semester_1/DEM/Final_Project_2/preprocessed'
apply_clahe(image_folder, output_folder)