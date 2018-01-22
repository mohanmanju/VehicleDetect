from keras.models import model_from_json
import numpy
import cv2
from keras.preprocessing.image import img_to_array
import time

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("weights.h5")
model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])


camera = cv2.VideoCapture(0)
retval, image = camera.read()


#uncomment for grey scale image
'''image = cv2.resize(im, (10, 10))
image = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY )
print(image)'''

#get part of the image
def section_image():
    count = 0
    testimage = []
    for rows in image:
        testimage.append(list(rows[:64]))
        count+=1
        if(count==64):
            break
    lst =[]
    lst.append(list(testimage))
    result = model.predict(numpy.asarray(lst))
    print(result)

while True:

    retval, im = camera.read()
    image = cv2.resize(im, (64, 64))
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame',gray)
    if cv2.waitKey(1)  == ord('q'):
        break
    image = img_to_array(image)
    lst =[]
    lst.append(list(image))
    result = model.predict(numpy.asarray(lst))
    print(result)
    time.sleep(0.1)
camera.release()
cv2.destroyAllWindows()
