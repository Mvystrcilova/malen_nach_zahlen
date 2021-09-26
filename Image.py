import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from Cluster import Pixel, Cluster, NormalPixel, BorderPixel
import queue


class Image:
    def __init__(self, file, min_cluster_size, num_of_colors, step_size):
        self.file = file
        self.image = None
        self.clustered_image = None
        self.color_labels = None
        self.min_cluster_size = min_cluster_size
        self.num_of_colors = num_of_colors
        self.color_dict = {}
        self.clusters = []
        self.pixel_array = None
        self.step_size = step_size

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
        for i in range(0, self.image.shape[0], self.step_size):
            for j in range(0, self.image.shape[1], self.step_size):
                if ((i + self.step_size) < self.image.shape[0]) and ((j + self.step_size) < self.image.shape[1]):
                    unique, counts = np.unique(self.color_labels[i:i + self.step_size, j:j + self.step_size],
                                               return_counts=True)
                    clustered_image[i:i + self.step_size, j:j + self.step_size] = self.color_dict[
                        unique[np.argmax(counts)]]
                elif (i + self.step_size) < self.image.shape[0]:
                    unique, counts = np.unique(self.color_labels[i:i + self.step_size, j:j + self.step_size],
                                               return_counts=True)
                    clustered_image[i:i + self.step_size, j:] = self.color_dict[unique[np.argmax(counts)]]
                else:
                    unique, counts = np.unique(self.color_labels[i:, j:j + self.step_size], return_counts=True)
                    clustered_image[i:, j:j + self.step_size] = self.color_dict[unique[np.argmax(counts)]]

        self.clustered_image = clustered_image
        plt.imshow(self.clustered_image / 255)
        plt.show()

    def create_no_border_clusters(self):
        self.get_no_border_pixel_array()
        cluster_id = 0
        cluster = Cluster(id=cluster_id)
        problematic_clusters = []
        q = queue.Queue()
        for i in range(self.pixel_array.shape[0]):
            for j in range(self.pixel_array.shape[1]):
                pixel = self.pixel_array[i, j]
                curr_cluster_color = pixel.color
                if pixel.cluster_id == -1:
                    q.put(pixel)
                    cluster.pixels.append(pixel)
                    cluster.color = curr_cluster_color
                    while q.qsize() > 0:
                        # print(q.qsize())
                        curr_pixel = q.get()
                        curr_pixel.cluster_id = cluster_id
                        neighbours = self.get_neighbours(curr_pixel.x, curr_pixel.y)
                        for neighbour in neighbours:
                            if neighbour is not None:
                                n = self.pixel_array[neighbour]
                                if (n.color == curr_cluster_color) and (not n.open):
                                    n.open = True
                                    q.put(n)
                                    cluster.pixels.append(n)
                    cluster.pixels = list(set(cluster.pixels))
                    if len(cluster.pixels) < self.min_cluster_size:
                        problematic_clusters.append(cluster)
                    self.clusters.append(cluster)
                    cluster_id += 1
                    cluster = Cluster(id=cluster_id)

        self.deal_with_small_clusters(problematic_clusters)
        self.show_no_borderpixel_borders()

    def get_neighbouring_clusters(self, cluster_id=-1, cluster=None):
        if cluster is None:
            cluster = self.get_cluster(cluster_id)
        neighboring_clusters = []
        for pixel in cluster.pixels:
            neighbours = self.get_neighbours(pixel.x, pixel.y)
            for neighbour in neighbours:
                if neighbour is not None:
                    n = self.pixel_array[neighbour]
                    if (n.cluster_id != pixel.cluster_id) and (n.cluster_id not in neighboring_clusters):
                        neighboring_clusters.append(n.cluster_id)
        return neighboring_clusters

    def deal_with_small_clusters(self, problematic_clusters):
        for cluster in problematic_clusters:
            neighbour_clusters = self.get_neighbouring_clusters(cluster=cluster)
            neighbour_clusters = [x for x in neighbour_clusters if x not in problematic_clusters]
            assert len(neighbour_clusters) > 0
            # 'does not necessary need to be bigger than zero,
            # TODO: let's just hope this never happens for now... haha

            closest_cluster_id = self.get_closest_color(cluster, neighbour_clusters)
            closest_cluster = self.get_cluster(closest_cluster_id)
            for pixel in cluster.pixels:
                pixel.cluster_id = closest_cluster_id
                pixel.color = closest_cluster.color
                closest_cluster.pixels.append(pixel)
            self.clusters.remove(cluster)

    def get_pixel(self, x, y):
        return self.pixel_array[x, y]

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

    def show_no_borderpixel_borders(self):
        img = np.ones(self.image.shape)
        border = []
        for i in range(self.pixel_array.shape[0]):
            for j in range(self.pixel_array.shape[1]):
                pixel = self.pixel_array[i, j]
                for n in self.get_neighbours(pixel.x, pixel.y):
                    if n is not None:
                        neighbour_pixel = self.pixel_array[n]
                        if (neighbour_pixel.cluster_id != pixel.cluster_id) and (neighbour_pixel not in border):
                            img[i, j] = [0, 0, 0]
                            border.append(pixel)
        plt.imshow(img)
        plt.show()

    def get_no_border_pixel_array(self):
        self.pixel_array = np.empty((self.image.shape[0], self.image.shape[1]), dtype=Pixel)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                pixel = NormalPixel(i, j, color=self.color_labels[i, j])
                self.pixel_array[i, j] = pixel

    def show_pixel_array_cluster(self, cluster_id):
        cluster_img = np.ones(self.image.shape)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                if self.pixel_array[i, j].cluster_id == cluster_id:
                    cluster_img[i, j, :] = self.color_dict[self.pixel_array[i, j].color] / 255
        plt.imshow(cluster_img)
        plt.show()

    def show_cluster(self, cluster_id):
        cluster_img = np.ones(self.image.shape)
        cluster = self.get_cluster(cluster_id)
        print(f'cluster id: {cluster.id}, cluster size: {cluster.get_cluster_size()}')
        for pixel in cluster.pixels:
            cluster_img[pixel.x, pixel.y] = self.color_dict[pixel.color] / 255
        # for border_pixel in cluster.border:
        #     cluster_img[border_pixel.x, border_pixel.y] = [0, 0, 0]
        plt.imshow(cluster_img)
        plt.show()

    def make_pre_merging_checkup(self):
        for cluster in self.clusters:
            for pixel in cluster.pixels:
                assert pixel.cluster_id == cluster.id
                assert pixel.color == cluster.color
                assert not pixel.border_pixel
            for pixel in cluster.border:
                # assert cluster.id in pixel.neighbour_clusters
                assert pixel.border_pixel

    def get_closest_color(self, cluster: Cluster, other_clusters: list[int]):
        distance = float('inf')
        closest_cluster_id = -1
        for other_c_id in other_clusters:
            other_cluster = self.get_cluster(other_c_id)
            color = self.color_dict[other_cluster.color]
            if np.linalg.norm(color - self.color_dict[cluster.color]) < distance:
                distance = np.linalg.norm(color - self.color_dict[cluster.color])
                closest_cluster_id = other_cluster.id
        return closest_cluster_id
