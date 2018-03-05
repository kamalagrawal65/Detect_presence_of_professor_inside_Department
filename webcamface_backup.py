import face_recognition
import cv2
import os
# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

faceEncodings=[]
for filename in os.listdir("../FRLibrary/trainingImages"):
    tmp_image = face_recognition.load_image_file(filename)
    tmp_face_encoding = face_recognition.face_encodings(tmp_image)[0]
    faceEncodings.append((filename,tmp_face_encoding))

#print(faceEncodings[0][1])

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
kamal_image = face_recognition.load_image_file("k.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]
kamal_face_encoding = face_recognition.face_encodings(kamal_image)[0]

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        match1 = face_recognition.compare_faces([obama_face_encoding], face_encoding)
        match2 = face_recognition.compare_faces([kamal_face_encoding], face_encoding)
        name = "Unknown"
        if match1[0]:
            name = "Barack"
        elif match2[0]:
            name = "Kamal"

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
