import time
from library import Lorex
from library import Grabber
from library import Utils


test_nr = 0

if test_nr == 0:
    grabber = Grabber.RTSPGrabber(3)
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
    for counter in range(1):
        print(counter)
        start = time.time()
        # Use detection_scale=0.5 for 4x speedup (1/2 resolution detection)
        # Use draw_grid=False to skip expensive grid drawing
        detections = camera.get_aruco(
            draw=False,
            world_undistort=False,
            detection_scale=0.5,
            draw_grid=False
        )
        detections = Lorex.parse_detections(detections)
        end = time.time()

        detections = camera.get_aruco(detection_scale=0.5, draw=True, world_undistort=False)
        detections = Lorex.parse_detections(detections)
        print(detections)

        print(f"Detection time: {(end - start) * 1000:.1f} ms")
        time.sleep(1)
    #detections = camera.get_aruco(draw=True, world_undistort=False)
    camera.stop()

if test_nr == 3:
    camera_name = 'tiger'
    camera = Lorex.LorexCamera(camera_name)
    # draw_grid=True by default, set to False for faster rendering
    image = camera.draw_board_axes_and_grid(undistort=True, draw_grid=True)
    camera.stop()
    Utils.show_full(image)
