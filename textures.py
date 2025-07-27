import pygame as pg
import moderngl as mgl


import pygame as pg
import moderngl as mgl


class Textures:
    def __init__(self, app):
        self.app = app
        self.ctx = app.ctx

        # List of filenames with one texture per face.
        # Update these file names as appropriate.
        # face_files = [
        #     "top.png",  # face_id 0
        #     "bottom.png",  # face_id 1
        #     "right.png",  # face_id 2
        #     "leftt.png",  # face_id 3
        #     "back.png",  # face_id 4
        #     "front.png",  # face_id 5
        # ]

        face_files = [
            "frameless.png",  # face_id 0
            "frameless.png",  # face_id 1
            "frameless.png",  # face_id 2
            "frameless.png",  # face_id 3
            "frameless.png",  # face_id 4
            "frameless.png",  # face_id 5
        ]

        self.face_textures = []
        for i, file in enumerate(face_files):
            tex = self.load(file)
            # Bind each to its corresponding texture unit (0 to 5)
            tex.use(location=i)
            self.face_textures.append(tex)

    def load(self, file_name):
        surface = pg.image.load(f"assets/{file_name}")
        # Flip texture vertically (adjust flip_x/flip_y as needed)
        surface = pg.transform.flip(surface, flip_x=True, flip_y=False)
        texture = self.ctx.texture(
            size=surface.get_size(),
            components=4,
            data=pg.image.tostring(surface, "RGBA", False),
        )
        texture.anisotropy = 32.0
        texture.build_mipmaps()
        texture.filter = (mgl.NEAREST, mgl.NEAREST)
        return texture
