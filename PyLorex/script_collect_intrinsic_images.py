import os
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts.warning=false")
from LorexLib import Utils
from LorexLib import Sound
from LorexLib import Settings
from LorexLib import Grabber
import easygui

# Configuration ######################

camera_name = 'tiger' # The ID of the camera
number_of_images = 20
#######################################

channel = Settings.channels[camera_name]
folders = Utils.get_calibration_paths(camera_name)
sounds = Sound.SoundPlayer()
camera = Grabber.RTSPGrabber(channel=channel)

print(f"[config] camera={camera_name}, channel={channel}, "
      f"target={number_of_images} images, "
      f"output={folders['calibration_images_folder']}")



calibration_images_folder = folders['calibration_images_folder']
folder_exists = Utils.check_exists(calibration_images_folder)
nr_existing_images = 0
response = False
if folder_exists:
    images = Utils.get_sorted_images(calibration_images_folder)
    nr_existing_images = len(images)
    message = f'Found {nr_existing_images} images. Delete?'
    response = easygui.ynbox(message)

if response or not folder_exists:
    Utils.create_folder(calibration_images_folder, clear=True)
    nr_existing_images = 0

start_index = nr_existing_images
end_index = start_index + number_of_images

for counter in range(start_index, end_index):
    file_name = f'frame{counter:02d}.jpg'
    image_file = os.path.join(calibration_images_folder, file_name)
    message = f'Capturing image {counter+1} of {end_index}'
    print(message)
    sounds.play('pips', volume=0.25)
    sounds.speak(message, volume=1.0)
    result = camera.save(path=image_file)
    sounds.play('shutter', volume=1.0)

print('Done.')
camera.stop()