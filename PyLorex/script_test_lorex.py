import time
from library import Lorex
from library import Grabber
from library import Utils
from library import ServerClient

test_nr = 2

channel_nr = 2
camera_name = 'tiger'
iterations = 3

if test_nr == 0:
    # Test low level frame grabbing
    grabber = Grabber.RTSPGrabber(channel_nr)
    f = grabber.show_latest_bgr()
    grabber.stop()

if test_nr == 1:
    # Test getting a single frame from the Lorex camera
    undistort = False
    camera = Lorex.LorexCamera(camera_name)
    frame = camera.get_frame(undistort=undistort)
    camera.stop()
    Utils.show_full(frame)

if test_nr == 2:
    # Test ArUco detection speed
    camera = Lorex.LorexCamera(camera_name)
    for counter in range(5):
        print(counter)
        start = time.time()
        detections = camera.get_aruco()
        detections = Lorex.parse_detections(detections)
        end = time.time()
        print(detections)
        print(f"Detection time: {(end - start) * 1000:.1f} ms")
        time.sleep(1)
    detections = camera.get_aruco(draw=True)
    print('Final detection with drawing:')
    print(detections)
    camera.stop()

if test_nr == 3:
    # Test drawing board axes and grid
    camera = Lorex.LorexCamera(camera_name)
    image = camera.draw_board_axes_and_grid(undistort=False, draw_grid=True)
    camera.stop()
    Utils.show_full(image)

if test_nr == 4:
    # Test telemetry client
    client = ServerClient.TelemetryClient()
    result = client.ping()
    print(result)
    cameras = client.list_cameras()
    print(cameras)
    for i in range(iterations):
        info = client.get_trackers()
        print(info)
        time.sleep(1)














