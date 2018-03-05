import face_recognition
import cv2
import os

# Issues
# Slow Running of Camera

# 0 for laptop and 1 for External Camera
video_capture = cv2.VideoCapture(1)

# Face Encoding has to be done for all the images present
faceEncodings={}
folder="./trainingImages"
for person in os.listdir(folder):
    name=person
    faceEncodings[name]=[]
    for filename in os.listdir(os.path.join(folder, person)):
        img_path = os.path.join(os.path.join(folder, person), filename)
        img_path = img_path.replace("\\","/")
        tmp_image = face_recognition.load_image_file(img_path)
        tmp_face_encoding = face_recognition.face_encodings(tmp_image)[0]
        faceEncodings[name].append(tmp_face_encoding)

# print(faceEncodings['Kamal'][0])

# Load a sample picture and learn how to recognize it.
while True:
    ret, frame = video_capture.read()
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        countNames={}

        # Count face encodings and counters
        for name in faceEncodings:
            countNames[name]=0
            for faces in faceEncodings[name]:
                match = face_recognition.compare_faces([faces], face_encoding)
                if match[0]:
                    countNames[name]+=1

        # Determine the name of the Person
        finalName = "Unknown"
        maxCount=0
        for name in countNames:
            if(countNames[name]>maxCount):
                maxCount=countNames[name]
                finalName=name

        #print((countNames,maxCount))

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, finalName, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
