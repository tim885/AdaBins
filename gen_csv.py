# generate csv file for train/test
import os
curr_dir = os.path.abspath(os.path.dirname(__file__))


def gen_test_csv():
    """gen image sets for train/test on 1449 labeled samples"""
    print('gen test set csv file')
    dataset_name = 'densenyuv2'
    imageset_abs_dir = os.path.join(curr_dir, 'train_test_inputs')
    print('input dataset data dir:{}'.format(imageset_abs_dir))

    scene_rgb_abs_dir = '/home/xuchong/Dataset/DenseDepthNYUDv2/data/nyu2_test'
    rgb_files = [fn for fn in os.listdir(scene_rgb_abs_dir) if 'colors.png' in fn]
    depth_files = [fn for fn in os.listdir(scene_rgb_abs_dir) if 'depth.png' in fn]
    rgb_files = sorted(rgb_files)
    depth_files = sorted(depth_files)

    out_csv_path = os.path.join(imageset_abs_dir, '{}_test_files_with_gt.txt'.format(dataset_name))
    with open(out_csv_path, 'w') as f:
        lines = ''
        for rgb_file, depth_file in zip(rgb_files, depth_files):
            lines += '{0} {1} 518.8579\n'.format(rgb_file, depth_file)
        f.writelines(lines[:-1])  # rm last \n
        print('write to csv file:{}'.format(out_csv_path))


if __name__ == '__main__':
    gen_test_csv()
