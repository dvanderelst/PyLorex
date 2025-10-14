from library import Lorex
from library import Utils
from library import Sound
from library import Settings
import os

# Configuration ######################
channel = 2 # Which RTSP channel to use
camera_name = 'tiger' # The ID of the camera
number_of_images = 20
#######################################


folders = Utils.get_calibration_paths(camera_name)
sounds = Sound.SoundPlayer(sound_folder='library/sounds')
camera = Lorex.LiveRTSPGrabber(channel=channel)

image_folder = folders['image_folder']
Utils.create_folder(image_folder)

for counter in range(number_of_images):
    print(f'Capturing image {counter:03d} of {number_of_images}')
    file_name = f'frame{counter:02d}.jpg'
    image_file = os.path.join(image_folder, file_name)
    sounds.play('pips', volume=0.1)
    result = camera.save(path=image_file)
    sounds.play('shutter')

print('Done.')