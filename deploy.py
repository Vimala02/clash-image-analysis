from flask import Flask, render_template, request, send_file, url_for
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from fpdf import FPDF
import tempfile


app = Flask(__name__)


# Path for saving processed images
STATIC_FOLDER = 'static/processed_images'
if not os.path.exists(STATIC_FOLDER):
    os.makedirs(STATIC_FOLDER)



# Function to load class names from a file
def load_class_names(class_file_path):
    if not os.path.exists(class_file_path):
        raise FileNotFoundError(f"Class file not found: {class_file_path}")
    with open(class_file_path, "r") as f:
        return f.read().strip().split("\n")


    

# Function to annotate the frame with bounding boxes, labels, and confidence
def annotate_frame(frame, detections, class_list, confidence_threshold=0.5):
    
    """Annotates the frame with bounding boxes, class labels, and confidence scores.
    Args:
        frame (np.array): The image/frame to annotate.
        detections (pd.DataFrame): Detected bounding boxes, confidence scores, and class IDs.
        class_list (list): List of class names.
        confidence_threshold (float): Threshold to switch between two colors.
    Returns:
        np.array: Annotated frame.
    """
    # Define two distinct colors
    color_high_confidence = (0, 255, 0)  # Green for high confidence
    color_low_confidence = (0, 0, 255)  # Red for low confidence

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = map(int, row[:4])
        confidence = row[4]
        class_id = int(row[5])

        # Get the class name
        class_name = class_list[class_id] if class_id < len(class_list) else f"ID:{class_id}"

        # Choose color based on confidence
        color = color_high_confidence if confidence >= confidence_threshold else color_low_confidence

        # Draw bounding box with thicker lines
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness=3)

        # Add semi-transparent background for text
        label = f"{class_name} ({confidence:.2f})"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        label_y1 = max(y1 - label_height - baseline, 0)
        label_y2 = y1
        cv2.rectangle(frame, (x1, label_y1), (x1 + label_width, label_y2), color, -1)

        # Add text on top of the box
        cv2.putText(frame, label, (x1, label_y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    return frame

# Function to generate a PDF with tabulated metadata
def generate_pdf_with_table(images, metadata, pdf_path):
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    for image_path, meta in zip(images, metadata):
        # Add a page for each image
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="YOLO Detection Results", ln=True, align="C")
        pdf.ln(10)

        # Add the image (absolute file path required)
        pdf.image(image_path, x=10, y=30, w=180)
        pdf.ln(180)  # Adjust the spacing below the image to move the table further down

        # Add metadata as a table
        if not meta.empty:
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt="Detection Metadata (Tabulated):", ln=True)
            pdf.ln(5)  # Add extra space above the table

            # Add table headers
            pdf.set_fill_color(200, 200, 200)
            pdf.cell(30, 10, "X1", border=1, align="C", fill=True)
            pdf.cell(30, 10, "Y1", border=1, align="C", fill=True)
            pdf.cell(30, 10, "X2", border=1, align="C", fill=True)
            pdf.cell(30, 10, "Y2", border=1, align="C", fill=True)
            pdf.cell(50, 10, "Class Name", border=1, align="C", fill=True)
            pdf.cell(30, 10, "Confidence", border=1, align="C", fill=True)
            pdf.ln()

            # Add table rows
            for _, row in meta.iterrows():
                pdf.cell(30, 10, str(int(row["x1"])), border=1, align="C")
                pdf.cell(30, 10, str(int(row["y1"])), border=1, align="C")
                pdf.cell(30, 10, str(int(row["x2"])), border=1, align="C")
                pdf.cell(30, 10, str(int(row["y2"])), border=1, align="C")
                pdf.cell(50, 10, str(row["class_name"]), border=1, align="C")
                pdf.cell(30, 10, f"{row['confidence']:.2f}", border=1, align="C")
                pdf.ln()
        else:
            pdf.set_font("Arial", size=10)
            pdf.cell(200, 10, txt="No detections found for this image.", ln=True)

    # Save the PDF
    pdf.output(pdf_path)
    return pdf_path

# Function to process and display results for multiple images
def process_images(model, uploaded_files, class_list, confidence_threshold):
    processed_images = []
    absolute_image_paths = []  # Paths for the PDF
    metadata_list = []

    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)  # Reset file pointer
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            continue

        # Perform inference
        results = model(frame, conf=confidence_threshold)

        # Extract detections
        detections = results[0].boxes.data.cpu().numpy() if results[0].boxes is not None else []
        detections_df = pd.DataFrame(detections, columns=["x1", "y1", "x2", "y2", "confidence", "class_id"])

        # Annotate frame with the updated function
        frame = annotate_frame(frame, detections_df, class_list, confidence_threshold)

        # Save processed image
        image_filename = f"processed_{uploaded_file.filename}"
        image_filepath = os.path.join(STATIC_FOLDER, image_filename)
        cv2.imwrite(image_filepath, frame)

        # Add absolute and URL paths
        absolute_image_paths.append(image_filepath)  # For PDF
        processed_images.append(url_for('static', filename=f'processed_images/{image_filename}'))  # For the browser

        # Store metadata
        if not detections_df.empty:
            detections_df["class_name"] = detections_df["class_id"].apply(
                lambda x: class_list[int(x)] if int(x) < len(class_list) else f"ID:{int(x)}"
            )
            metadata_list.append(detections_df)
        else:
            metadata_list.append(pd.DataFrame())

    return processed_images, absolute_image_paths, metadata_list

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get form inputs
        uploaded_files = request.files.getlist("images")
        model_path = request.form["model_path"]
        class_file_path = request.form["class_file_path"]
        confidence_threshold = float(request.form["confidence_threshold"])

        try:
            if not os.path.exists(model_path):
                return "Model file not found", 404

            model = YOLO(model_path)
            class_list = load_class_names(class_file_path)

            processed_images, absolute_image_paths, metadata_list = process_images(
                model, uploaded_files, class_list, confidence_threshold
            )

            # Generate PDF with absolute image paths
            pdf_path = os.path.join(tempfile.gettempdir(), "yolo_detection_results.pdf")
            generate_pdf_with_table(absolute_image_paths, metadata_list, pdf_path)

            # Render results
            return render_template("index.html", processed_images=processed_images, pdf_path=pdf_path)

        except Exception as e:
            return str(e), 500

    return render_template("index.html", processed_images=[], pdf_path=None)

@app.route("/download_pdf")
def download_pdf():
    pdf_path = request.args.get("pdf_path")
    return send_file(pdf_path, as_attachment=True, download_name="yolo_detection_results.pdf")

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0', port=8000)


