import numpy as np
from midvoxio.voxio import vox_to_arr

TARGET_PATH = "nyan_sakura"


def load_image(imagePath: str):
    voxel_tensor = vox_to_arr(imagePath)
    return voxel_tensor.astype(np.float32)


## Target Voxel Generation (Grows Upwards)
# target_voxel = load_image(f"assets/voxel_models/{TARGET_PATH}.vox")
# target_voxel = np.swapaxes(target_voxel, 1, 2)
# num_frames = target_voxel.shape[1]
# target_state = np.zeros((num_frames, *target_voxel.shape))
# for i in range(0, num_frames):
#     frame = np.zeros(target_voxel.shape)
#     frame[:, : i + 1, :, :] = target_voxel[:, : i + 1, :, :]
#     target_state[i] = frame

## Target Voxel Masking
target_voxel = load_image(f"assets/voxel_models/small_sakura.vox")

target_voxel = np.swapaxes(target_voxel, 1, 2)
nca_gif_state = np.load(f"state/static/{TARGET_PATH}.npy")  # frames,xyz, channels

target_voxel = target_voxel[tuple(slice(0, s) for s in nca_gif_state[0].shape)]

num_steps = nca_gif_state.shape[0]
inv_mask = target_voxel[..., 3] <= 0  # xyz
inv_mask = np.repeat(np.expand_dims(inv_mask, axis=0), num_steps, axis=0)  # frames,xyz
nca_gif_state[inv_mask, :] = 0
nca_gif_state[:, ..., :3] = target_voxel[..., :3]


np.save(f"state/static/{TARGET_PATH}_masked.npy", nca_gif_state[15:, ...])
