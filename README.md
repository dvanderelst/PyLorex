# PyLorex

PyLorex is a Python library designed to interface with Lorex security cameras, allowing users to access camera feeds, control camera settings, and manage recordings programmatically.

The library allows for **intrinsic camera calibration** and **homography estimation**, enabling users to correct lens distortion and map camera views to real-world coordinates: With intrinsics and a homography, every pixel can be mapped to an (X, Y) position on that plane. But if the object is not on that plane, the mapping doesn’t hold anymore. Also, this only holds as long **as the camera is not moved**.

# Scripts

+ `script_collect_calibration_images`: Captures a sequence of calibration images from a configured camera and saves them to the project's calibration image folder.
+ `script_run_calibration`: Calibrates a camera from a set of chessboard images and saves the camera intrinsics and an undistorted preview image.
+ `script_assess_calibration`: Assesses the quality of a camera calibration. Generates an HTML report.
+ `script_test_lorex`: Tests getting images from a Lorex camera and undistorting them using previously saved camera intrinsics.