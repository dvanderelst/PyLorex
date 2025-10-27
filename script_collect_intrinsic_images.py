import os
from library import Utils
from library import Sound
from library import Settings
from library import Grabber


# Configuration ######################

camera_name = 'tiger' # The ID of the camera
number_of_images = 5
#######################################

channel = Settings.channels[camera_name]
folders = Utils.get_calibration_paths(camera_name)
sounds = Sound.SoundPlayer(sound_folder='library/sounds')
camera = Grabber.RTSPGrabber(channel=channel)

calibration_images_folder = folders['calibration_images_folder']
Utils.create_folder(calibration_images_folder)

for counter in range(number_of_images):
    print(f'Capturing image {counter:03d} of {number_of_images}')
    file_name = f'frame{counter:02d}.jpg'
    image_file = os.path.join(calibration_images_folder, file_name)
    sounds.play('pips', volume=0.1)
    result = camera.save(path=image_file)
    sounds.play('shutter')

print('Done.')
camera.stop()