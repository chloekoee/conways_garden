from constants.settings import *


class VoxelHandler:
    def __init__(self, scene):
        self.app = scene.app
        self.nca = scene.nca

        # Ray casting result
        self.voxel_id = None
        self.target_found = False
        self.voxel_position = None

        ## this may be used for building/adding blocks
        self.voxel_normal = glm.ivec3(0)

    def remove_voxel(self):
        if self.target_found:
            print(f"removing {self.voxel_position}")
            x, y, z = self.voxel_position
            ## Remove voxel by setting its alpha channel to 0 if it is filled
            if self.nca.state[x, y, z, 3] > 0:
                self.nca.state[x, y, z, 3] = 0
                self.nca.delete_voxel(x, y, z)  ## update the texture representation
                self.nca.mesh.rebuild()
            self.target_found = False

    def update(self):
        self.ray_cast()

    def is_filled(self, position):
        x, y, z = position
        nca_tensor = self.nca.state
        x_dim, y_dim, z_dim = nca_tensor[..., 3].shape
        if (0 <= x < x_dim and 0 <= y < y_dim and 0 <= z < z_dim) and nca_tensor[
            x, y, z, 3
        ] > 0:  # 0.1
            return True
        return False

    def ray_cast(self):
        x1, y1, z1 = self.app.player.position
        x2, y2, z2 = self.app.player.position + self.app.player.forward * MAX_RAY_DIST

        target = glm.ivec3(x1, y1, z1)
        self.voxel_normal = glm.ivec3(0)
        step_dir = -1

        dx = glm.sign(x2 - x1)
        delta_x = min(dx / (x2 - x1), 10000000.0) if dx != 0 else 10000000.0
        max_x = delta_x * (1.0 - glm.fract(x1)) if dx > 0 else delta_x * glm.fract(x1)

        dy = glm.sign(y2 - y1)
        delta_y = min(dy / (y2 - y1), 10000000.0) if dy != 0 else 10000000.0
        max_y = delta_y * (1.0 - glm.fract(y1)) if dy > 0 else delta_y * glm.fract(y1)

        dz = glm.sign(z2 - z1)
        delta_z = min(dz / (z2 - z1), 10000000.0) if dz != 0 else 10000000.0
        max_z = delta_z * (1.0 - glm.fract(z1)) if dz > 0 else delta_z * glm.fract(z1)

        while not (max_x > 1.0 and max_y > 1.0 and max_z > 1.0):
            if self.is_filled(target):
                if step_dir == 0:
                    self.voxel_normal.x = -dx
                elif step_dir == 1:
                    self.voxel_normal.y = -dy
                else:
                    self.voxel_normal.z = -dz
                self.target_found = True
                self.voxel_position = glm.ivec3(target)
                return

            if max_x < max_y:
                if max_x < max_z:
                    target.x += dx
                    max_x += delta_x
                    step_dir = 0
                else:
                    target.z += dz
                    max_z += delta_z
                    step_dir = 2
            else:
                if max_y < max_z:
                    target.y += dy
                    max_y += delta_y
                    step_dir = 1
                else:
                    target.z += dz
                    max_z += delta_z
                    step_dir = 2

        self.target_found = False
