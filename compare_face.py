import face_recognition

known_image = face_recognition.load_image_file("k.jpg")
# Load in our image of a group of people
unknown_image = face_recognition.load_image_file("Kamal.jpg")

# Create a biden encoding
biden_encoding = face_recognition.face_encodings(known_image)[0]
# create an encoding based off our group photo
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare the encodings and try to determine if Biden exists within a photo
results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
# Print the results
print(results)
