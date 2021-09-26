import random


class Pixel:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class NormalPixel(Pixel):
    def __init__(self, x, y, color):
        super().__init__(x, y)
        self.border_pixel = False
        self.color = color
        self.cluster_id = -1
        self.open = False
        self.border_distance = -1

    def assign_cluster(self, cluster_id):
        self.cluster_id = cluster_id


class Cluster:
    def __init__(self, id: int):
        self.color = -1
        self.id = id
        self.pixels = []
        self.border = []
        self.cluster_size = -1

    def get_cluster_size(self):
        return len(self.pixels)

