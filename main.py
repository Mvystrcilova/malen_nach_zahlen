from Image import Image
home = '/Users/m_vys/PycharmProjects/malen_nach_zahlen'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_name = f'{home}/pictures/person.jpg'
    img = Image(file_name, 1000, 10)
    img.read_file()
    img.cluster_colors()
    img.color_image()
    img.create_clusters()
    # img.show_cluster(0)
    img.show_cluster(1)
    img.show_cluster(2)
    img.show_cluster(6)
    img.show_cluster(7)
    img.show_cluster(10)
    print(len(img.clusters))
    # img.merge_small_clusters()
    # img.show_cluster(1)
    # img.show_cluster(2)
    # img.show_cluster(6)
    # img.show_cluster(10)
    # print(len(img.clusters))
    # img.show_cluster(11)
    # img.show_cluster(12)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
