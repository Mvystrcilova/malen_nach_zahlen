import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.cluster import KMeans
from Cluster import Pixel, Cluster, NormalPixel, BorderPixel
import queue


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
        self.pixel_array = None

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
        self.get_pixel_array()
        cluster_id = 0
        q = queue.Queue()
        for i in range(self.pixel_array.shape[0]):
            for j in range(self.pixel_array.shape[1]):
                pixel = self.pixel_array[i, j]
                if not pixel.border_pixel:
                    if pixel.cluster_id == -1:
                        q.put(pixel)
                        while q.qsize() > 0:
                            curr_pixel = q.get()
                            curr_pixel.cluster_id = cluster_id
                            neighbours = self.get_neighbours(curr_pixel.x, curr_pixel.y)
                            for neighbour in neighbours:
                                if neighbour is not None:
                                    n = self.pixel_array[neighbour]
                                    if (not n.border_pixel) and (n.cluster_id == -1) and (not n.open):
                                        n.open = True
                                        q.put(n)
                        cluster_id += 1

                else:
                    pixel.cluster_id = -2
        self.show_borders()
        self.show_pixel_array_cluster(1)
        self.show_pixel_array_cluster(2)
        self.show_pixel_array_cluster(10)
        self.set_neighbour_clusters()
        print(len(self.clusters))

    def get_neighbour_clusters(self, border_pixel):
        neighbours = self.get_neighbours(border_pixel.x, border_pixel.y)
        neighbouring_clusters = []
        for neighbour_coords in neighbours:
            if neighbour_coords is not None:
                neighbour_pixel = self.pixel_array[neighbour_coords]
                if not neighbour_pixel.border_pixel:
                    neighbouring_clusters.append(neighbour_pixel.cluster_id)
        return set(neighbouring_clusters)

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

    def show_borders(self):
        img = np.ones(self.image.shape)
        for i in range(self.pixel_array.shape[0]):
            for j in range(self.pixel_array.shape[1]):
                if self.pixel_array[i, j].border_pixel:
                    img[i, j, :] = [0, 0, 0]
        plt.imshow(img)
        plt.show()

    def set_neighbour_clusters(self):
        for i in range(self.pixel_array.shape[0]):
            for j in range(self.pixel_array.shape[1]):
                pixel = self.pixel_array[i, j]
                if not pixel.border_pixel:
                    cluster = self.get_cluster(pixel.cluster_id)
                    if cluster is None:
                        cluster = Cluster(id=pixel.cluster_id)
                        cluster.color = pixel.color
                        self.clusters.append(cluster)
                    cluster.pixels.append(pixel)

        for i in range(self.pixel_array.shape[0]):
            for j in range(self.pixel_array.shape[1]):
                pixel = self.pixel_array[i, j]
                if (i == 355) and (j == 178):
                    print('stop')
                if pixel.border_pixel:
                    neighbours = self.get_neighbours(pixel.x, pixel.y)
                    for n in neighbours:
                        if n is not None:
                            n_pixel = self.pixel_array[n]
                            if (i == 355) and (j == 178):
                                print(
                                    f'neighbour: {n_pixel.x}, {n_pixel.y}, border pixel: {n_pixel.border_pixel}, cluster id: {n_pixel.cluster_id}')
                            if not n_pixel.border_pixel:
                                cluster = self.get_cluster(n_pixel.cluster_id)
                                assert cluster is not None
                                cluster.border.append(pixel)

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                pixel = self.pixel_array[i, j]
                if not pixel.border_pixel:
                    assert pixel.cluster_id != -1
                    assert pixel.color != -1

        print(len(self.clusters))

    def get_pixel_array(self):
        self.pixel_array = np.empty((self.image.shape[0], self.image.shape[1]), dtype=Pixel)
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                neighbours = self.get_neighbours(i, j)
                border_pixel = False
                for neighbour in neighbours:
                    if neighbour is not None:
                        if (self.color_labels[neighbour] != self.color_labels[i, j]) and (self.color_labels[neighbour] != -1):
                            border_pixel = True
                            self.color_labels[i, j] = -1
                            break
                if not border_pixel:
                    pixel = NormalPixel(i, j, color=self.color_labels[i, j])
                else:
                    pixel = BorderPixel(i, j)
                self.pixel_array[i, j] = pixel
        self.show_borders()

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
        for border_pixel in cluster.border:
            cluster_img[border_pixel.x, border_pixel.y] = [0, 0, 0]
        plt.imshow(cluster_img)
        plt.show()

    def deal_with_no_feasible_border_cluster(self, cluster):
        neighbouring_cluster_ids = []
        for border_pixel in cluster.border:
            neighbours = self.get_neighbours(border_pixel.x, border_pixel.y)
            for n in neighbours:
                if n is not None:
                    pixel = self.pixel_array[n]
                    if pixel.border_pixel:
                        for n_cluster_id in self.get_neighbour_clusters(pixel):
                            if n_cluster_id != cluster.id:
                                neighbouring_cluster_ids.append(n_cluster_id)
        neighbouring_cluster_ids = set(neighbouring_cluster_ids)
        print(f'{len(neighbouring_cluster_ids)} neighbouring clusters: {neighbouring_cluster_ids} for no feasible border cluster: {cluster.id}')
        closest_cluster_id = self.get_closest_color(cluster, neighbouring_cluster_ids)
        closest_cluster = self.get_cluster(closest_cluster_id)
        for pixel in cluster.pixels:
            pixel.cluster_id = closest_cluster_id
            pixel.color = closest_cluster.color
            closest_cluster.pixels.append(pixel)
        for border_pixel in cluster.border:
            self.change_border_pixel_to_normal(border_pixel, cluster, closest_cluster)

        self.remove_cluster_from_pixel_array(cluster.id)
        self.clusters.remove(cluster)

    def merge_small_clusters(self):
        i = 0
        changed = False
        clusters_to_remove = []
        old_clusters = self.clusters.copy()
        for cluster in old_clusters:
            print('cluster id:', cluster.id, ' cluster size: ', cluster.get_cluster_size())
            if cluster.get_cluster_size() < self.min_cluster_size:
                feasible_border_pixels = [pixel for pixel in cluster.border if len(self.get_neighbour_clusters(pixel)) > 1]
                # feasible_border_pixels = [pixel for pixel in feasible_border_pixels if self.get_cluster(
                # pixel.cluster_id).get_cluster_size() > self.min_cluster_size]
                assert len(feasible_border_pixels) > 0

                selected_border_cluster = -1
                distance = float('inf')
                for pixel in feasible_border_pixels:
                    for id in self.get_neighbour_clusters(pixel):
                        if id != cluster.id:
                            if np.linalg.norm(self.color_dict[cluster.color] - self.color_dict[
                                self.get_cluster(id).color]) < distance:
                                distance = np.linalg.norm(
                                    self.color_dict[cluster.color] - self.color_dict[self.get_cluster(id).color])
                                selected_border_cluster = id

                print('selecting cluster id: ', selected_border_cluster, ' to merge with cluster id: ', cluster.id)
                cluster_to_merge = self.get_cluster(selected_border_cluster)

                for pixel in cluster.pixels:
                    pixel.cluster_id = cluster_to_merge.id
                    pixel.color = cluster_to_merge.color
                    cluster_to_merge.pixels.append(pixel)

                for i, border_pixel in enumerate(cluster.border):
                    border_pixel_neighbours = self.get_neighbour_clusters(border_pixel)
                    if border_pixel_neighbours == {cluster_to_merge.id}:
                        self.change_border_pixel_to_normal(border_pixel, cluster, cluster_to_merge)
                    else:
                        cluster_to_merge.border.append(border_pixel)
                    # print(i)
                self.remove_cluster_from_pixel_array(cluster.id)
            print(cluster.id, 'done')
            self.show_borders()

    def make_pre_merging_checkup(self):
        for cluster in self.clusters:
            for pixel in cluster.pixels:
                assert pixel.cluster_id == cluster.id
                assert pixel.color == cluster.color
                assert not pixel.border_pixel
            for pixel in cluster.border:
                # assert cluster.id in pixel.neighbour_clusters
                assert pixel.border_pixel

    def get_closest_color(self, cluster, other_clusters):
        distance = float('inf')
        closest_cluster_id = -1
        for other_c_id in other_clusters:
            other_cluster = self.get_cluster(other_c_id)
            color = self.color_dict[other_cluster.color]
            if np.linalg.norm(color - self.color_dict[cluster.color]) < distance:
                distance = np.linalg.norm(color - self.color_dict[cluster.color])
                closest_cluster_id = other_cluster.id
        return closest_cluster_id

    def change_border_pixel_to_normal(self, border_pixel, cluster, cluster_to_merge):
        new_pixel = NormalPixel(border_pixel.x, border_pixel.y, cluster_to_merge.color)
        new_pixel.cluster_id = cluster_to_merge.id
        # print(f'changed pixel {border_pixel.x}, {border_pixel.y} from border to normal')
        self.pixel_array[border_pixel.x, border_pixel.y] = new_pixel
        cluster_to_merge.pixels.append(new_pixel)

        for c in self.get_neighbour_clusters(border_pixel):
            cluster_to_remove_border = self.get_cluster(c)
            if border_pixel in cluster_to_remove_border.border:
                cluster_to_remove_border.border.remove(border_pixel)

    def remove_cluster_from_pixel_array(self, cluster_to_remove_id):
        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                pixel = self.pixel_array[i, j]
                if pixel.border_pixel:
                    pass
                else:
                    assert pixel.cluster_id != cluster_to_remove_id

