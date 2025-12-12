import cv2
import numpy as np
import tensorflow as tf
import warnings
from keras.models import load_model
import absl.logging
from tkinter import *
from tkinter import PhotoImage
import sys  # For exiting the program

absl.logging.set_verbosity(absl.logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="keras")

# Load the updated model
model = load_model(r"D:\CSC582 PROJECT NEW\Emotion_detection_with_CNN-main\model\final_emotion_model.keras")

# Initialize the face detector
faceDetect = cv2.CascadeClassifier(r"D:\CSC583 PROJECT CODE\Facial Emotion\haarcascade_frontalface_default.xml")

# Updated emotion labels (Removed "Fear" and "Disgust")
labels_dict = {0: 'angry', 1: 'disgust', 2: 'happy', 3: 'neutral', 4: 'surprise'}

# Variable to track whether the app is exiting
is_exiting = False

def start_emotion_detection():
    """Start the facial emotion detection."""
    global is_exiting
    video = cv2.VideoCapture(0)  
    if not video.isOpened():
        print("Error: Could not open the camera.")
        return

    while not is_exiting:  # Check if the app is exiting
        ret, frame = video.read()
        if not ret:
            print("Error: Unable to read from the camera.")
            break

        # Get screen dimensions
        screen_width = window.winfo_width()
        screen_height = window.winfo_height()

        # Resize the video feed to fill the screen
        frame = cv2.resize(frame, (screen_width, screen_height))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)

        for x, y, w, h in faces:
            sub_face_img = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_img, (48, 48))
            normalize = resized / 255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]

            # Draw rectangles and labels
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Show video feed
        cv2.imshow("Emotion Detection", frame)

        k = cv2.waitKey(1)
        if k == ord('x'):  # If 'x' is pressed, break the loop
            print("Closing the detector...")
            break

    # Release resources when the loop ends
    video.release()
    cv2.destroyAllWindows()

def exit_app():
    """Stop the camera feed and exit the application."""
    global is_exiting
    is_exiting = True  # Set the flag to true to stop the camera feed
    cv2.destroyAllWindows()  # Close any OpenCV windows
    window.destroy()  # Destroy the Tkinter window
    sys.exit()  # Exit the program completely

# Tkinter GUI for starting the camera
window = Tk()
window.title("Facial Emotion Recognition")
window.geometry("800x600")  # Initial size of the window
window.attributes('-fullscreen', True)  # Set to fullscreen mode

def exit_fullscreen(event=None):
    """Exit fullscreen mode when pressing ESC."""
    window.attributes('-fullscreen', False)

# Bind ESC key to exit fullscreen
window.bind("<Escape>", exit_fullscreen)

# Add a background image to the window
bg_image = PhotoImage(file=r"C:\Users\Tengku Maria\Downloads\photo_2021-09-23_17-26-35-1920x1080.png")  # Ensure you have a .png file for the background
bg_label = Label(window, image=bg_image)
bg_label.place(relwidth=1, relheight=1)  # Cover the entire window

# Add a title above the "Start Camera" button
title_label = Label(window, text="Emotion Detection System", font=("Georgia", 52, "bold"), bg="#fbb6d0", fg="#d5006d")
title_label.place(relx=0.5, rely=0.3, anchor=CENTER)

# Center the start button
start_button = Button(window, text="Start Camera", command=start_emotion_detection)
start_button.config(font=("Georgia", 16, "bold"), bg="#fbb6d0", fg="#d5006d")
start_button.place(relx=0.5, rely=0.5, anchor=CENTER)

# Exit button below the start button
exit_button = Button(window, text="Exit", command=exit_app)
exit_button.config(font=("Georgia", 16, "bold"), bg="red", fg="white")
exit_button.place(relx=0.5, rely=0.7, anchor=CENTER)

window.mainloop()
