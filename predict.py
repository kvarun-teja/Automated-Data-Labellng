from ultralytics import YOLOWorld

# Initialize a YOLO-World model
model = YOLOWorld("/media/varunteja/Professional/Northeastern/Semester_1/DEM/Final_Project_2/runs/detect/train3/weights/best.pt")  # or select yolov8m/l-world.pt for different sizes

results = model.predict("/media/varunteja/Professional/Northeastern/Semester_1/DEM/Final_Project_2/_132346170_firstbusglasgowbusfleet.jpg")

# Show results
results[0].show()