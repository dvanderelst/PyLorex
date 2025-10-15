from pathlib import Path
import os
import shutil

def get_calibration_paths(camera_name):
    camera_name = str(camera_name)
    root_folder = 'calibration'
    image_folder = os.path.join(root_folder, f"images_{camera_name}")
    result_folder = os.path.join(root_folder, "results")
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

def create_folder(folder_path, clear=True):
    folder = Path(folder_path)
    if folder.exists():
        if clear:
            # Empty the folder
            for item in folder.glob('*'):
                if item.is_file(): item.unlink()
                else: shutil.rmtree(item)
    else: folder.mkdir(parents=True)

