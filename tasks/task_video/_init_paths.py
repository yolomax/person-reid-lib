import os
import os.path as osp
import sys


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)


this_dir = os.path.split(os.path.realpath(__file__))[0]

# add some packages that are not in the docker env.
py_package_path = osp.join(this_dir, '..', '..', '..','opt','mypython')
if osp.exists(py_package_path):
    add_path(py_package_path)

# add the lib
lib_path = osp.join(this_dir, '..', '..')
add_path(lib_path)

