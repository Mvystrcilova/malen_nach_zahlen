import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from Cluster import Pixel, Cluster, NormalPixel, BorderPixel
import queue
import random

class Image:
    def __init__(self, file, min_cluster_size, num_of_colors):
        self.file = file
        self.image = None
        self.clustered_image = None
        self.color_labels = None
        self.min_cluster_size = min_cluster_size
        self.num_of_colors = num_of_colors
        self.color_dict = {}
        self.clusters = []

    def read_file(self):
        image = cv2.imread(self.file)
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.imshow(self.image)
        plt.show()

    def cluster_colors(self):
        img_array = self.image.reshape((self.image.shape[1] * self.image.shape[0]), -1)
        k_means = KMeans(n_clusters=self.num_of_colors)
        result = k_means.fit(img_array)
        colors = result.cluster_centers_
        self.color_dict = {number: color for number, color in zip(np.arange(len(colors)), colors)}
        plt.pie([1 / self.num_of_colors] * self.num_of_colors, colors=np.array(colors / 255),
                labels=np.arange(len(colors)))
        plt.show()
        self.color_labels = result.labels_.reshape(self.image.shape[:2])

    def color_image(self):
        clustered_image = np.empty((self.image.shape[0], self.image.shape[1], 3))
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                clustered_image[i, j] = self.color_dict[self.color_labels[i, j]]
        self.clustered_image = clustered_image
        fig = plt.imshow(self.clustered_image / 255)
        plt.show()

    def create_clusters(self):
        pixel_array = self.get_pixel_array()
        cluster_id = 0
        q = queue.Queue()
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                pixel = pixel_array[i, j]
                if not pixel.border_pixel:
                    if pixel.cluster_id == -1:
                        q.put(pixel)
                        while q.qsize() > 0:
                            curr_pixel = q.get()
                            curr_pixel.cluster_id = cluster_id
                            neighbours = self.get_neighbours(curr_pixel.x, curr_pixel.y)
                            for neighbour in neighbours:
                                if neighbour is not None:
                                    n = pixel_array[neighbour]
                                    if (not n.border_pixel) and (n.cluster_id == -1) and (not n.open):
                                        n.open = True
                                        q.put(n)
                        cluster_id += 1

                else:
                    pixel.cluster_id = -2
        self.show_borders(pixel_array)
        self.show_pixel_array_cluster(1, pixel_array)
        self.show_pixel_array_cluster(2, pixel_array)
        self.show_pixel_array_cluster(10, pixel_array)
        self.set_neighbour_clusters(pixel_array)
        print(len(self.clusters))

    def get_pixel(self, pixel_array, x, y):
        return pixel_array[x, y]

    def get_upper_neighbour(self, x, y):
        if y > 0:
            return x, y - 1

    def get_right_neighbour(self, x, y):
        if x > 0:
            return x - 1, y

    def get_left_neighbour(self, x, y):
        if x < (self.clustered_image.shape[0] - 1):
            return x + 1, y

    def get_lower_neighbour(self, x, y):
        if y < (self.clustered_image.shape[1] - 1):
            return x, y + 1

    def get_neighbours(self, x, y):
        neighbours = self.get_right_neighbour(x, y), self.get_left_neighbour(x, y), self.get_upper_neighbour(x,
                                                                                                             y), self.get_lower_neighbour(
            x, y)
        return neighbours

    def get_cluster(self, cluster_id):
        for cluster in self.clusters:
            if cluster.id == cluster_id:
                return cluster
        return None

    def get_colors(self):
        return self.color_dict.values()

    def show_borders(self, pixel_array):
        img = np.ones(self.image.shape)
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                if pixel_array[i, j].border_pixel:
                    img[i, j, :] = [0, 0, 0]
        plt.imshow(img)
        plt.show()

    def set_neighbour_clusters(self, pixel_array):
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                pixel = pixel_array[i, j]
                if not pixel.border_pixel:
                    cluster = self.get_cluster(pixel.cluster_id)
                    if cluster is None:
                        cluster = Cluster(id=pixel.cluster_id)
                        cluster.color = pixel.color
                        self.clusters.append(cluster)
                    cluster.pixels.append(pixel)
        
        for i in range(pixel_array.shape[0]):
            for j in range(pixel_array.shape[1]):
                pixel = pixel_array[i, j]

                if pixel.border_pixel:
                    neighbours = self.get_neighbours(pixel.x, pixel.y)
                    for neighbour in neighbours:
                        if neighbour is not None:
                            neighbour_pixel = pixel_array[neighbour]

                            if not neighbour_pixel.border_pixel:
                                if neighbour_pixel.cluster_id not in pixel.neighbour_clusters:
                                    pixel.neighbour_clusters.append(neighbour_pixel.cluster_id)
                                cluster = self.get_cluster(neighbour_pixel.cluster_id)
                                assert cluster is not None
                                cluster.border.append(pixel)

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                pixel = pixel_array[i, j]
                if not pixel.border_pixel:
                    assert pixel.cluster_id != -1
        print(len(self.clusters))

    def get_pixel_array(self):
        pixel_array = np.empty((self.image.shape[0], self.image.shape[1]), dtype=Pixel)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                neighbours = self.get_neighbours(i, j)
                border_pixel = False
                for neighbour in neighbours:
                    if (neighbour is None) or (self.color_labels[neighbour] != self.color_labels[i, j]):
                        border_pixel = True
                        break
                if not border_pixel:
                    pixel = NormalPixel(i, j, color=self.color_labels[i, j])
                else:
                    pixel = BorderPixel(i, j)
                pixel_array[i, j] = pixel
        self.show_borders(pixel_array)

        return pixel_array

    def show_pixel_array_cluster(self, cluster_id, pixel_array):
        cluster_img = np.ones(self.image.shape)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if pixel_array[i, j].cluster_id == cluster_id:
                    cluster_img[i, j, :] = self.color_dict[pixel_array[i, j].color]/255
        plt.imshow(cluster_img)
        plt.show()

    def show_cluster(self, cluster_id):
        cluster_img = np.ones(self.image.shape)
        cluster = self.get_cluster(cluster_id)
        for pixel in cluster.pixels:
            cluster_img[pixel.x, pixel.y] = self.color_dict[pixel.color]/255
        for border_pixel in cluster.border:
            cluster_img[border_pixel.x, border_pixel.y] = [0, 0, 0]
        plt.imshow(cluster_img)
        plt.show()

    def merge_small_clusters(self):
        for cluster in self.clusters:
            print('cluster id:', cluster.id, ' cluster size: ', cluster.get_cluster_size())
            if (cluster.get_cluster_size() < self.min_cluster_size) and (cluster.id != 10):

                feasible_border_pixels = [pixel for pixel in cluster.border if len(pixel.neighbour_clusters) > 1]
                if len(feasible_border_pixels) == 0:
                    break

                rnd = random.randint(0, len(feasible_border_pixels)-1)
                border_pixel = feasible_border_pixels[rnd]
                border_pixel.neighbour_clusters.remove(cluster.id)

                rnd_cluster_to_merge = random.randint(0, len(border_pixel.neighbour_clusters)-1)
                print('selecting cluster id: ', border_pixel.neighbour_clusters[rnd_cluster_to_merge], ' to merge with cluster id: ', cluster.id)
                cluster_to_merge = self.get_cluster(border_pixel.neighbour_clusters[rnd_cluster_to_merge])

                for pixel in cluster.pixels:
                    pixel.cluster_id = cluster_to_merge.id
                    pixel.color = cluster_to_merge.color
                    cluster_to_merge.pixels.append(pixel)

                for i, border_pixel in enumerate(cluster.border):
                    if (len(border_pixel.neighbour_clusters) == 2) and (cluster.id in border_pixel.neighbour_clusters) and (cluster_to_merge.id in border_pixel.neighbour_clusters):
                        new_pixel = NormalPixel(border_pixel.x, border_pixel.y, cluster_to_merge.color)
                        new_pixel.cluster_id = cluster_to_merge.id
                        cluster_to_merge.pixels.append(new_pixel)
                        border_pixel.neighbour_clusters.remove(cluster.id)
                        for c in border_pixel.neighbour_clusters:
                            cluster_to_remove_border = self.get_cluster(c)
                            cluster_to_remove_border.border.remove(border_pixel)
                    else:
                        cluster_to_merge.border.append(border_pixel)
                        if cluster.id in border_pixel.neighbour_clusters:
                            border_pixel.neighbour_clusters.remove(cluster.id)
                    # print(i)
                self.clusters.remove(cluster)
            print(cluster.id, 'done')
