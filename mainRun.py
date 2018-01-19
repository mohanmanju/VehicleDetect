import cv2
from keras.preprocessing.image import img_to_array
import glob
from trainVehicle import Network


network = Network()
network.build_model()
network.compile_model()
network.read_data()
network.train()
