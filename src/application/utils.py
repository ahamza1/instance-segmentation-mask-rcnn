import os
import uuid


def get_tmp_file_name(file_name):
    return f"{uuid.uuid1()}{os.path.splitext(file_name)[1]}"


def get_tmp_file_path(dir_path, file_name):
    return os.path.join(dir_path, file_name)
