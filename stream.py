from flask import Flask, render_template, Response
from flask_socketio import SocketIO
import cv2
import numpy as np
import face_recognition
import os
import threading
import queue

# Flask app
app = Flask(__name__)
# SocketIO app
socketioApp = SocketIO(app)

# Queue for frames
face_images_path="faces"
face_images=[]
face_names=[]
# Get all the images in the faces folder
image_files = [f for f in os.listdir(face_images_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Define some constants
SCALE_FACTOR = 0.25
UNKNOWN_THRESHOLD = 0.6

# Get all the images in the faces folder
for cl in image_files:
    # Skip if not image
    if not cl.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    # Load image
    current_image = face_recognition.load_image_file(f'{face_images_path}/{cl}')
    # Skip if image is empty
    if current_image is None or current_image.size == 0:
        print(f"Unable to read image: {cl}")
        continue
    # Resize image
    face_images.append(current_image)
    # Get name only
    face_names.append(os.path.splitext(cl)[0])

# Function that convert the face distance calculated during face recognition process (i.e., face_distance) into a confidence score
def face_distance_to_conf(face_distance, face_match_threshold=0.6):
    # Below is a simple linear function that converts the distance to a confidence score
    if face_distance > face_match_threshold:
        range = (1.0 - face_match_threshold)
        linear_val = (1.0 - face_distance) / (range * 2.0)
        return linear_val
    # Otherwise we linearly interpolate the confidence to a more logarithmic scale
    else:
        range = face_match_threshold
        linear_val = 1.0 - (face_distance / (range * 2.0))
        return linear_val + ((1.0 - linear_val) * ((linear_val - 0.5) * 2) ** 0.2)

# Function that computes the face encodings for these known images.
def find_face_encodings(images):
    face_encodings_list =[]
    for img in images:
        # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
        img= cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # Get the face encodings for each face in each image file
        encode=face_recognition.face_encodings(img)[0]
        # Add face encoding for current image with corresponding label (name) to the training data
        face_encodings_list.append(encode)
    # Return the array of face encodings    
    return face_encodings_list

# Detecting faces in a video frame and comparing them with known face encodings
def process_frame(img):
    # Skip if image is empty
    if img is None or img.size == 0:
        return img
    # Resize image
    resized_frame = cv2.resize(img, (0, 0), None, SCALE_FACTOR, SCALE_FACTOR)
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    current_frame_faces = face_recognition.face_locations(resized_frame)
    current_frame_encodings = face_recognition.face_encodings(resized_frame, current_frame_faces)

    # Loop over each face found in the frame to see if it's someone we know.
    for face_encoding, face_coordinates in zip(current_frame_encodings, current_frame_faces):
        # See if the face is a match for the known face(s)
        face_matches = face_recognition.compare_faces(face_encodings_list_known, face_encoding)
        # Use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(face_encodings_list_known, face_encoding)
        # Get the best match index
        best_match_index = np.argmin(face_distances)

        # If there is a match
        if face_matches[best_match_index]:
            # Get the name to display on the video
            matched_name = face_names[best_match_index].upper()
            # Calculate the match percentage
            matchPerc = round(face_distance_to_conf(face_distances[best_match_index]) * 100)
        else:
            # Set the name to display on the video
            matched_name = "Unknown"
            # Set the match percentage to 0
            matchPerc = 0

        y1, x2, y2, x1 = face_coordinates
        y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, f'{matched_name} {matchPerc}%', (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

    return img

# Get the face encodings for each face in each image file
face_encodings_list_known= find_face_encodings(face_images)
print("encoding complete!")

# video capture 
camera = cv2.VideoCapture(0)

# Thread to read frames from camera
class CameraThread(threading.Thread):
    def __init__(self, camera, frame_queue):
        threading.Thread.__init__(self)
        self.camera = camera
        self.frame_queue = frame_queue
        self.stop_flag = False

    def run(self):
        frame_counter = 0
        while not self.stop_flag:
            success, img = self.camera.read()
            if not success:
                break
            self.frame_queue.put(img)

    def stop(self):
        self.stop_flag = True

# class ProcessThread(threading.Thread):
#     def __init__(self, frame_queue, processed_queue):
#         threading.Thread.__init__(self)
#         self.frame_queue = frame_queue
#         self.processed_queue = processed_queue
#         self.stop_flag = False

#     def run(self):
#         while not self.stop_flag:
#             img = self.frame_queue.get()
#             img = process_frame(img)
#             self.processed_queue.put(img)

#     def stop(self):
#         self.stop_flag = True

# Thread to process frames
class ProcessThread(threading.Thread):
    def __init__(self, frame_queue, processed_queue):
        threading.Thread.__init__(self)
        self.frame_queue = frame_queue
        self.processed_queue = processed_queue
        self.stop_flag = False

    def run(self):
        frame_counter = 0
        while not self.stop_flag:
            img = self.frame_queue.get()
            if frame_counter % 2 == 0:  # Only process every other frame
                img = process_frame(img)
            self.processed_queue.put(img)
            frame_counter += 1  # Increment the frame counter

    def stop(self):
        self.stop_flag = True

#Thread to display frames
class DisplayThread(threading.Thread):
    def __init__(self, processed_queue):
        threading.Thread.__init__(self)
        self.processed_queue = processed_queue
        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            img = self.processed_queue.get()
            cv2.imshow("Webcam", img)
            if cv2.waitKey(1) == ord('q'):
                break
        camera_thread.stop()
        process_thread.stop()
        camera.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.stop_flag = True

# Create queues
frame_queue = queue.Queue(maxsize=5)
processed_queue = queue.Queue(maxsize=5)

# Create new threads
camera_thread = CameraThread(camera, frame_queue)
process_thread = ProcessThread(frame_queue, processed_queue)
display_thread = DisplayThread(processed_queue)

# Start new Threads
camera_thread.start()
process_thread.start()
# display_thread.start()

# gen_frames() function generates frames to be displayed on the webpage
def gen_frames():
    while True:
        # Capture frame-by-frame
        img = processed_queue.get()
        # encode OpenCV raw frame to jpg and displaying it
        ret, buffer = cv2.imencode('.jpg', img)
        # convert the image to bytes and yield as output
        img = buffer.tobytes()
        #stream video frames one by one
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

# route for video streaming
@app.route('/video_feed')
def video_feed():
    # return the response generated along with the specific media
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# route for home page
@app.route('/')
def index():
    #Streaming Page
    return render_template('index.html')

def run():
    socketioApp.run(app)

if __name__ == '__main__':
    socketioApp.run(app)