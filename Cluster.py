import random

class Pixel:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class BorderPixel(Pixel):
    def __init__(self, x, y):
        super().__init__(x, y)
        self.border_pixel = True
        self.neighbour_clusters = []


class NormalPixel(Pixel):
    def __init__(self, x, y, color):
        super().__init__(x, y)
        self.border_pixel = False
        self.color = color
        self.cluster_id = -1
        self.open = False

    def assign_cluster(self, cluster_id):
        self.cluster_id = cluster_id


class Cluster:
    def __init__(self, id: int):
        self.color = -1
        self.id = id
        self.pixels = []
        self.border = []
        self.cluster_size = -1

    def get_marker_coordinate(self):
        rnd = random.randint(0, len(self.pixels))
        rnd_pixel = self.pixels[rnd]

    def get_closes_border(self, pixel):
        pass

    def get_cluster_size(self):
        return len(self.pixels)
