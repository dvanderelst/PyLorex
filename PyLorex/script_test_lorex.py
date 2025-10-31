import time
from library import Lorex
from library import Grabber
from library import Utils

test_nr = 2

if test_nr == 0:
    grabber = Grabber.RTSPGrabber(2)
    f = grabber.show_latest_bgr()
    grabber.stop()

if test_nr == 1:
    camera_name = 'tiger'
    undistort = False
    camera = Lorex.LorexCamera(camera_name)
    frame = camera.get_frame(undistort=undistort)
    camera.stop()
    Utils.show_full(frame)

if test_nr == 2:
    camera_name = 'tiger'
    camera = Lorex.LorexCamera(camera_name)
    for counter in range(5):
        print(counter)
        start = time.time()
        detections = camera.get_aruco(draw=False, world_undistort=False)
        detections = Lorex.parse_detections(detections)
        print(detections)
        end = time.time()
        print(f"Detection time: {(end - start) * 1000:.1f} ms")
        time.sleep(1)
    camera.stop()

if test_nr == 3:
    camera_name = 'tiger'
    camera = Lorex.LorexCamera(camera_name)
    image = camera.draw_board_axes_and_grid(undistort=True)
    camera.stop()
    Utils.show_full(image)
