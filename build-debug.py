import os

if __name__ == '__main__':
    os.system('cd build && cmake -DCMAKE_BUILD_TYPE=Debug ..')
    os.system('cd build && make -j 16')
