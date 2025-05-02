from settings import *
from voxel_handler import VoxelHandler
from world_objects.chunk import Chunk

"""
Class holding the world's terrain -  comprised of several chunks
"""

class World:
    def __init__(self, app):
        self.app = app
        self.chunks = [None for _ in range(WORLD_VOL)]
        self.voxels = np.empty([WORLD_VOL, CHUNK_VOL], dtype = 'uint8')
        self.build_chunks()
        self.build_chunk_mesh()
        self.voxel_handler = VoxelHandler(self)

    def update(self):
        self.voxel_handler.update()

    def build_chunks(self):
        for x in range(WORLD_W):
            for y in range(WORLD_H):
                for z in range(WORLD_D):
                    chunk = Chunk(self, position=(x,y,z))
                    chunk_index = x + WORLD_W * z + WORLD_AREA * y

                    # store pointer to chunk itself 
                    self.chunks[chunk_index] = chunk

                    # store world's chunk's voxels
                    self.voxels[chunk_index] = chunk.build_voxels()
                    
                    # store chunks own voxels
                    chunk.voxels = self.voxels[chunk_index]

    def build_chunk_mesh(self):
        for chunk in self.chunks:
            chunk.build_mesh()

    def render(self):
        for chunk in self.chunks:
            chunk.render()