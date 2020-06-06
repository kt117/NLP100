import os


def make_file():
    for i in range(10):
        path = "chapter{}".format(str(i + 1).zfill(2))

        try:
            os.mkdir(path)
        except FileExistsError:
            pass

        for j in range(10):
            file_path = path + "/{}.py".format(str(10 * i + j).zfill(2))
            if not os.path.isfile(file_path):
                with open(file_path, mode='w') as f:
                    f.write("")


make_file()