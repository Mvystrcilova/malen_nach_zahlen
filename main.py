from Image import Image
home = '/Users/m_vys/PycharmProjects/malen_nach_zahlen'

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file_name = f'{home}/pictures/person.jpg'
    img = Image(file_name, 100, 5, step_size=1)
    img.read_file()
    img.cluster_colors()
    img.color_image()
    img.create_no_border_clusters()
    img.make_pre_merging_checkup()
    for cluster in img.clusters:
        img.show_cluster(cluster.id)
    print(len(img.clusters))
    print(len(img.clusters))

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
