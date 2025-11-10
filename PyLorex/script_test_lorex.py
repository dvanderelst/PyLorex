import time
from library import Lorex
from library import Grabber
from library import Utils
from library import ServerClient

test_nr = 4

channel_nr = 2
camera_name = 'tiger'
iterations = 10

if test_nr == 0:
    grabber = Grabber.RTSPGrabber(channel_nr)
    f = grabber.show_latest_bgr()
    grabber.stop()

if test_nr == 1:
    undistort = False
    camera = Lorex.LorexCamera(camera_name)
    frame = camera.get_frame(undistort=undistort)
    camera.stop()
    Utils.show_full(frame)

if test_nr == 2:
    camera = Lorex.LorexCamera(camera_name)
    for counter in range(1):
        print(counter)
        start = time.time()
        detections = camera.get_aruco()
        detections = Lorex.parse_detections(detections)
        end = time.time()
        print(detections)
        print(f"Detection time: {(end - start) * 1000:.1f} ms")
        time.sleep(1)
    camera.stop()

if test_nr == 3:
    camera = Lorex.LorexCamera(camera_name)
    image = camera.draw_board_axes_and_grid(undistort=True, draw_grid=True)
    camera.stop()
    Utils.show_full(image)

if test_nr == 4:
    client = ServerClient.TelemetryClient()
    cameras = client.list_cameras()
    info = client.ping()
    print(cameras)
    print(info)


