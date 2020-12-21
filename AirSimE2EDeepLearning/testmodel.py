import tensorflow as tf
from keras.models import load_model
import sys
import numpy as np
import glob
import os
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
    
#tf.config.gpu.set_per_process_memory_fraction(0.75)
#tf.config.gpu.set_per_process_memory_growth(True)

if ('../../PythonClient/' not in sys.path):
    sys.path.insert(0, '../../PythonClient/')

airsim_version=1.3
if airsim_version>=1.3:
    from airsim import *
else:
    #new airsim version, or stay with local copy of old airsim version
    from AirSimClient import *

# << Set this to the path of the model >>
# If None, then the model with the lowest validation loss from training will be used
MODEL_PATH = None

if (MODEL_PATH == None):
    models = glob.glob('model/models/*.h5') 
    best_model = max(models, key=os.path.getctime)
    MODEL_PATH = best_model
    
print('Using model {0} for testing.'.format(MODEL_PATH))
model = load_model(MODEL_PATH)


client = CarClient()
#client = CarClient(ip="192.168.86.98")
#client = CarClient(ip="192.168.1.169")

client.confirmConnection()
client.enableApiControl(True)

#car_controls = CarControls() # airsim old version, local copy
car_controls = CarControls() #airsim 1.3

print('Connection established!')
car_controls.steering = 0
car_controls.throttle = 0
car_controls.brake = 0
    # go forward
car_controls.throttle = -0.5
car_controls.steering = 0
client.setCarControls(car_controls)
print("Go Forward now")
#time.sleep(3)   # let car drive a bit
image_buf = np.zeros((1, 59, 255, 3))
state_buf = np.zeros((1,4))
def get_image():
    if airsim_version >= 1.3:
        imgreq = ImageRequest(0, ImageType.Scene, False, False)
    else:
        imgreq = ImageRequest(0, AirSimImageType.Scene, False, False) 
    image_response = client.simGetImages([imgreq])[0]
    image1d = np.fromstring(image_response.image_data_uint8, dtype=np.uint8)
    print(image_response.height, image_response.width)
    if airsim_version >= 1.3:
        image_rgba = image1d.reshape(image_response.height, image_response.width, 3)
    else:
        image_rgba = image1d.reshape(image_response.height, image_response.width, 4)
        
    return image_rgba[76:135,0:255,0:3].astype(float)
while (True):
    car_state = client.getCarState()
    
    if (car_state.speed < 5):
        car_controls.throttle = 1.0
    else:
        car_controls.throttle = 0.0
    
    image_buf[0] = get_image()
    state_buf[0] = np.array([car_controls.steering, car_controls.throttle, car_controls.brake, car_state.speed])
    model_output = model.predict([image_buf, state_buf])
    car_controls.steering = round(0.5 * float(model_output[0][0]), 2)
    
    print('Sending steering = {0}, throttle = {1}'.format(car_controls.steering, car_controls.throttle))
    
    client.setCarControls(car_controls)
