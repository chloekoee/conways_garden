import numpy as np
from midvoxio.voxio import vox_to_arr

NCA_PATH = "small_sakura"


def load_image(imagePath: str):
    voxel_tensor = vox_to_arr(imagePath)
    return voxel_tensor.astype(np.float32)


target_voxel = load_image(f"assets/voxel_models/{NCA_PATH}.vox")
target_voxel = np.swapaxes(target_voxel, 1, 2)
num_frames = target_voxel.shape[1]
state = np.zeros((num_frames, *target_voxel.shape))
for i in range(0, num_frames):
    frame = np.zeros(target_voxel.shape)
    frame[:, : i + 1, :, :] = target_voxel[:, : i + 1, :, :]
    state[i] = frame

np.save(f"state/static/{NCA_PATH}.npy", state)
