from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def count_faces(image_path):
    # Load the image
    image = cv2.imread(image_path)

    try:
        # Extract faces using DeepFace
        result = DeepFace.extract_faces(image_path, detector_backend='mtcnn')

        # Check the structure of the result for the 'mtcnn' backend
        if isinstance(result, list) and all(isinstance(face.get('face'), np.ndarray) for face in result):
            # If the result is a list of dictionaries with 'face' key
            face_locations = [(int(face['facial_area']['y']), int(face['facial_area']['x'] + face['facial_area']['w']),
                                int(face['facial_area']['y'] + face['facial_area']['h']),
                                int(face['facial_area']['x'])) for face in result]
        else:
            raise ValueError("Unexpected structure in the result. Unable to determine face locations.")

        # Count the number of faces
        num_faces = len(face_locations)

        return num_faces, face_locations

    except Exception as e:
        print(f"Error in count_faces: {e}")
        print("Result:", result)
        raise  # Re-raise the exception for further debugging

def visualize_faces(image, face_locations):
    plt.imshow(image)
    ax = plt.gca()

    for face_location in face_locations:
        top, right, bottom, left = face_location
        rect = patches.Rectangle((left, top), right - left, bottom - top, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    plt.show()

def main():
    # Path to the image file
    image_path = "image/pic.jpg"  # Use the correct path separator for your operating system

    # Count faces in the image and get face locations
    try:
        num_faces, face_locations = count_faces(image_path)
        print(f"Number of faces in the image: {num_faces}")

        # Print face locations for debugging
        print("Face Locations:", face_locations)

        # Visualize the detected faces
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        visualize_faces(image_rgb, face_locations)

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
