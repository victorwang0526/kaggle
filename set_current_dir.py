import os


def set_current_dir(dir_name):
    print(os.listdir('./'))
    root_dir = os.getcwd()
    current_dir = os.path.join(root_dir, dir_name)
    os.chdir(current_dir)
    print('current dir: ', os.getcwd())

