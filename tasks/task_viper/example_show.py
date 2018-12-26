from lib.utils.example_show import SearchShow
from lib.utils.util import file_abs_path

if __name__ == '__main__':
    cur_dir = file_abs_path(__file__)
    file_path = cur_dir / 'output/log/search_result.h5'
    test = SearchShow(file_path)
    for i in range(10):
        test(test.false_example, 0, i)

