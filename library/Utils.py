from pathlib import Path
from library import Settings
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

    images_glob = []
    if os.path.exists(image_folder): images_glob = os.path.join(image_folder, "*.jpg")

    results = { 'root_folder': root_folder,
                'image_folder': image_folder,
                'result_folder': result_folder,
                'calibration_images': images_glob,
                'intrinsics_yml': intrinsics_yml,
                'undistorted_preview': undistorted_preview,
                'intrinsics_report': intrinsics_report }
    return results


def create_folder(folder_path, clear=True):
    folder = Path(folder_path)
    if folder.exists():
        if clear:
            # Empty the folder
            for item in folder.glob('*'):
                if item.is_file(): item.unlink()
                else: shutil.rmtree(item)
    else: folder.mkdir(parents=True)

