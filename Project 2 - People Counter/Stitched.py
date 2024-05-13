import cv2
import os

script_directory = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(script_directory, "unknown_customers")

# List of image file names in the input folder
image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Initialize a list of images
imgs = []

# Load and resize images
for image_file in image_files:
    image_path = os.path.join(input_folder, image_file)
    img = cv2.imread(image_path)
    img = cv2.resize(img, (0, 0), fx=0.4, fy=0.4)  # You can adjust the scaling factor if needed
    imgs.append(img)

# Show the original pictures
for i, img in enumerate(imgs):
    cv2.imshow(f"Image {i + 1}", img)

# Create a Stitcher
stitcher = cv2.Stitcher_create()  # Use cv2.Stitcher_create instead of cv2.Stitcher.create()

# Perform stitching
status, result = stitcher.stitch(imgs)

# Check if stitching is successful
if status != cv2.Stitcher_OK:
    print("Stitching failed")
else:
    print("Your Panorama is ready!!!")

    # Create the output folder if it doesn't exist
    output_folder = "output_panorama"
    os.makedirs(output_folder, exist_ok=True)

    # Path to save the final panorama image
    output_path = os.path.join(output_folder, "panorama.jpg")

    # Save the panorama image
    cv2.imwrite(output_path, result)
    print(f"Panorama image saved to {output_path}")

# Close all OpenCV windows
cv2.destroyAllWindows()
