import streamlit as st
import cv2
import numpy as np
from PIL import Image
import random
import tempfile
from ultralytics import YOLO, YOLOWorld  

# App title
st.title("Object Detection and Segmentation")

# Sidebar
st.sidebar.title("Options")
class_option = st.sidebar.radio("Choose Mode:", ['Detection', 'Segmentation'])

# Function to preprocess
def preprocess_image(image_array):
    # Convert to grayscale
    gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE - Preprocess
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

    return enhanced_bgr

if class_option == 'Detection':
    # Options for Detection
    detection_mode = st.sidebar.radio("Detection Mode:", ['With Labels', 'Without Labels'])

    custom_classes = []
    if detection_mode == 'With Labels':
        # User-Input for classes
        custom_classes_input = st.sidebar.text_input("Enter Classes (comma-separated)", "person, backpack, ball, cat, building, football")
        custom_classes = [cls.strip() for cls in custom_classes_input.split(",")]  # Clean class names

        
        model = YOLO('yolov8x-worldv2.pt')
        #Get input classes
        model.set_classes(custom_classes)
    else:
        # Initialize YOLOWorld model 
        model = YOLOWorld(r'runs/detect/train3/weights/best.pt')

    # Colors for detected objects
    def generate_colors(num_classes):
        return [tuple(random.randint(0, 255) for _ in range(3)) for _ in range(num_classes)]

    
    if custom_classes:
        class_colors = {cls: color for cls, color in zip(custom_classes, generate_colors(len(custom_classes)))}
    else:
        class_colors = {}

    # Function to annotate images
    def annotate_image(image, results):
        boxes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            cls_id = int(box.cls[0])
            class_name = results[0].names[cls_id]
            label = f"{class_name} {conf:.2f}"
            color = class_colors.get(class_name, (255, 255, 255))

            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # bounding box in YOLO format
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            width = x2 - x1
            height = y2 - y1
            x_center /= image.shape[1]
            y_center /= image.shape[0]
            width /= image.shape[1]
            height /= image.shape[0]
            boxes.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

        return image, boxes

    # Image upload and processing
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Preprocess
        preprocessed_image = preprocess_image(image_cv)

        # copy of the preprocessed image
        image_for_annotation = preprocessed_image.copy()

        # YOLO object detection
        results = model.predict(preprocessed_image)

        annotated_image, boxes = annotate_image(image_for_annotation, results)

        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB), caption="Preprocessed Image", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Detected Objects", use_container_width=True)

        # Save the preprocessed image
        preprocessed_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
        cv2.imwrite(preprocessed_image_path, preprocessed_image)

        # Save the annotated image
        annotated_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
        cv2.imwrite(annotated_image_path, annotated_image)

    
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            with open(preprocessed_image_path, "rb") as img_file:
                st.download_button(
                    label="Download Preprocessed Image",
                    data=img_file,
                    file_name="preprocessed_image.jpg",
                    mime="image/jpeg"
                )
        with button_col2:
            with open(annotated_image_path, "rb") as img_file:
                st.download_button(
                    label="Download Annotated Image",
                    data=img_file,
                    file_name="annotated_image.jpg",
                    mime="image/jpeg"
                )

        # Save bounding coordinates
        bbox_text = "\n".join(boxes)
        st.download_button(
            label="Download YOLO Bounding Box Coordinates",
            data=bbox_text,
            file_name="bounding_boxes.txt",
            mime="text/plain"
        )


    # Video upload and processing
    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        video_cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
        out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 20.0,
                            (int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        all_boxes = []

        while video_cap.isOpened():
            ret, frame = video_cap.read()
            if not ret:
                break

            # Preprocess each frame
            preprocessed_frame = preprocess_image(frame)

            # YOLO object detection
            results = model.predict(preprocessed_frame)

            annotated_frame, boxes = annotate_image(preprocessed_frame.copy(), results)
            all_boxes.extend(boxes)
            out.write(annotated_frame)
            stframe.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

        video_cap.release()
        out.release()

        bbox_text = "\n".join(all_boxes)

    
        button_col1, button_col2 = st.columns(2)
        with button_col1:
            st.download_button(
                label="Download YOLO Bounding Box Coordinates for Video",
                data=bbox_text,
                file_name="video_bounding_boxes.txt",
                mime="text/plain"
            )
        with button_col2:
            with open(output_video_path, "rb") as video_file:
                st.download_button(
                    label="Download Annotated Video",
                    data=video_file,
                    file_name="annotated_video.mp4",
                    mime="video/mp4"
                )

#Segmentation
elif class_option == 'Segmentation':
    # Initialize YOLO model for segmentation
    model = YOLO(r"yolo11x-seg.pt")

    # Image upload for segmentation
    uploaded_image = st.file_uploader("Upload an image for segmentation", type=["jpg", "jpeg", "png"])
    if uploaded_image:
        image = Image.open(uploaded_image).convert('RGB')
        image_array = np.array(image)
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)

        # Preprocess the image
        preprocessed_image = preprocess_image(image_bgr)

        # YOLO segmentation
        results = model.predict(preprocessed_image)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(preprocessed_image, cv2.COLOR_BGR2RGB), caption="Preprocessed Image", use_container_width=True)
        with col2:
            segmented_image = results[0].plot()  
            st.image(segmented_image, caption="Segmented Image", use_container_width=True)

    
        segmented_image_path = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg').name
        cv2.imwrite(segmented_image_path, cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR))

        # Save a combined mask with all classes
        if hasattr(results[0], 'masks') and results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy() 
            combined_mask = np.zeros_like(masks[0], dtype=np.uint8)  

            # Combine
            for mask in masks:
                combined_mask = np.maximum(combined_mask, (mask * 255).astype(np.uint8)) 

            # Save
            combined_mask_path = tempfile.NamedTemporaryFile(delete=False, suffix='_combined_mask.png').name
            cv2.imwrite(combined_mask_path, combined_mask)

            
            button_col1, button_col2 = st.columns(2)
            with button_col1:
                with open(segmented_image_path, "rb") as img_file:
                    st.download_button(
                        label="Download Segmented Image",
                        data=img_file,
                        file_name="segmented_image.jpg",
                        mime="image/jpeg"
                    )
            with button_col2:
                with open(combined_mask_path, "rb") as mask_file:
                    st.download_button(
                        label="Download Combined Mask",
                        data=mask_file,
                        file_name="combined_mask.png",
                        mime="image/png"
                    )
        else:
            st.warning("No ground truth masks found in the results.")




