from pathlib import Path
import os
import shutil

def get_calibration_paths(camera_name):
    camera_name = str(camera_name)
    root_folder = 'calibration'
    image_folder = os.path.join(root_folder, f"images_{camera_name}")
    result_folder = os.path.join(root_folder, "results")
    # Ensure results folder exists so callers can immediately write files
    os.makedirs(result_folder, exist_ok=True)
    # Existing, stable outputs
    intrinsics_yml      = os.path.join(result_folder, f"intrinsics_{camera_name}.yml")
    intrinsics_report   = os.path.join(result_folder, f"intrinsics_{camera_name}.html")
    undistorted_preview = os.path.join(result_folder, f"undistorted_{camera_name}.jpg")
    homography_report   = os.path.join(result_folder, f"homography_{camera_name}.html")
    # Timestamped filenames for per-run images
    suffix = f"_{camera_name}"
    raw_image_path      = os.path.join(result_folder, f"raw{suffix}.png")
    rectified_image_path= os.path.join(result_folder, f"rectified{suffix}.png")
    # Stable filenames for pose dumps (overwritten each run)
    pose_json_path      = os.path.join(result_folder, f"pose_{camera_name}.json")
    pose_npz_path       = os.path.join(result_folder, f"pose_{camera_name}.npz")
    images_glob = []
    if os.path.exists(image_folder): images_glob = os.path.join(image_folder, "*.jpg")
    return {
        'root_folder': root_folder,
        'image_folder': image_folder,
        'result_folder': result_folder,
        'calibration_images': images_glob,
        'intrinsics_yml': intrinsics_yml,
        'undistorted_preview': undistorted_preview,
        'intrinsics_report': intrinsics_report,
        'homography_report': homography_report,
        'raw_image': raw_image_path,
        'rectified_image': rectified_image_path,
        'pose_json': pose_json_path,
        'pose_npz': pose_npz_path,
    }



def create_folder(folder_path, clear=True):
    folder = Path(folder_path)
    if folder.exists():
        if clear:
            # Empty the folder
            for item in folder.glob('*'):
                if item.is_file(): item.unlink()
                else: shutil.rmtree(item)
    else: folder.mkdir(parents=True)

