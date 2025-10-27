from pathlib import Path
import os
import shutil
import glob
from natsort import natsorted





def get_calibration_paths(camera_name):
    camera_name = str(camera_name)
    root_folder = 'Calibration'
    calibration_images_folder = os.path.join(root_folder, f"Calibration_images_{camera_name}")
    result_folder = os.path.join(root_folder, "Results")
    intrinsics_yml = os.path.join(result_folder, f"intrinsics_{camera_name}.yml")
    intrinsics_report = os.path.join(result_folder, f"intrinsics_{camera_name}.html")
    undistorted_preview = os.path.join(result_folder, f"undistorted_{camera_name}.jpg")
    homography_report = os.path.join(result_folder, f"homography_{camera_name}.html")
    pose_json = os.path.join(result_folder, f"pose_{camera_name}.json")
    pose_npz = os.path.join(result_folder, f"pose_{camera_name}.npz")
    raw_image = os.path.join(result_folder, f"raw_{camera_name}.jpg")
    rectified_image = os.path.join(result_folder, f"rectified_{camera_name}.jpg")
    axes_overlay = os.path.join(result_folder, f"axes_overlay_{camera_name}.jpg")  # ← new

    return dict(
        result_folder=result_folder,
        calibration_images_folder = calibration_images_folder,
        intrinsics_yml=intrinsics_yml,
        intrinsics_report=intrinsics_report,
        undistorted_preview=undistorted_preview,
        homography_report=homography_report,
        pose_json=pose_json,
        pose_npz=pose_npz,
        raw_image=raw_image,
        rectified_image=rectified_image,
        axes_overlay=axes_overlay,
    )


def check_exists(folder_path):
    folder = Path(folder_path)
    return folder.exists()


def create_folder(folder_path, clear=True):
    folder = Path(folder_path)
    if folder.exists():
        if clear:
            for item in folder.glob('*'):
                if item.is_file(): item.unlink()
                else: shutil.rmtree(item)
    else: folder.mkdir(parents=True)


def get_sorted_images(folder_path, recursive=False):
    extensions = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff')
    files = []
    for ext in extensions: files.extend(glob.glob(f"{folder_path}/{ext}", recursive=recursive))
    return natsorted(files)