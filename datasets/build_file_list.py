
import argparse
import os
import glob
import random
import fnmatch

def parse_directory(path, rgb_prefix='img_', flow_x_prefix='flow_x_', flow_y_prefix='flow_y_'):
    """
    Parse directories holding extracted frames from standard benchmarks
    """
    print('parse frames under folder {}'.format(path))
    frame_folders = glob.glob(os.path.join(path, '*'))

    def count_files(directory, prefix_list):
        lst = os.listdir(directory)
        cnt_list = [len(fnmatch.filter(lst, x+'*')) for x in prefix_list]
        return cnt_list

    rgb_counts = {}
    flow_counts = {}
    for i,f in enumerate(frame_folders):
        all_cnt = count_files(f, (rgb_prefix, flow_x_prefix, flow_y_prefix))
        k = f.split('/')[-1]
        rgb_counts[k] = all_cnt[0]

        x_cnt = all_cnt[1]
        y_cnt = all_cnt[2]
        if x_cnt != y_cnt:
            raise ValueError('x and y direction have different number of flow images. video: '+f)
        flow_counts[k] = x_cnt
        if i % 200 == 0:
            print('{} videos parsed'.format(i))

    print('frame folder analysis done')
    return rgb_counts, flow_counts


def build_split_list(split_tuple, frame_info, split_idx, shuffle=False):
    split = split_tuple[split_idx]

    def build_set_list(set_list):
        rgb_list, flow_list = list(), list()
        for item in set_list:
            rgb_cnt = frame_info[0][item[0]]
            flow_cnt = frame_info[1][item[0]]
            rgb_list.append('{} {} {}\n'.format(item[0], rgb_cnt, item[1]))
            flow_list.append('{} {} {}\n'.format(item[0], flow_cnt, item[1]))
        if shuffle:
            random.shuffle(rgb_list)
            random.shuffle(flow_list)
        return rgb_list, flow_list

    train_rgb_list, train_flow_list = build_set_list(split[0])
    test_rgb_list, test_flow_list = build_set_list(split[1])
    return (train_rgb_list, test_rgb_list), (train_flow_list, test_flow_list)


def parse_ucf101_splits():
    class_ind = [x.strip().split() for x in open('ucf101_splits/classInd.txt')]
    class_mapping = {x[1]:int(x[0])-1 for x in class_ind}

    def line2rec(line):
        items = line.strip().split('/')
        label = class_mapping[items[0]]
        vid = items[1].split('.')[0]
        return vid, label

    splits = []
    for i in xrange(1, 4):
        train_list = [line2rec(x) for x in open('ucf101_splits/trainlist{:02d}.txt'.format(i))]
        test_list = [line2rec(x) for x in open('ucf101_splits/testlist{:02d}.txt'.format(i))]
        splits.append((train_list, test_list))
    return splits


def parse_hmdb51_splits():
    # load split file
    class_files = glob.glob('hmdb51_splits/*split*.txt')

    # load class list
    class_list = [x.strip() for x in open('hmdb51_splits/class_list.txt')]
    class_dict = {x: i for i, x in enumerate(class_list)}

    def parse_class_file(filename):
        # parse filename parts
        filename_parts = filename.split('/')[-1][:-4].split('_')
        split_id = int(filename_parts[-1][-1])
        class_name = '_'.join(filename_parts[:-2])

        # parse class file contents
        contents = [x.strip().split() for x in open(filename).readlines()]
        train_videos = [ln[0][:-4] for ln in contents if ln[1] == '1']
        test_videos = [ln[0][:-4] for ln in contents if ln[1] == '2']

        return class_name, split_id, train_videos, test_videos

    class_info_list = map(parse_class_file, class_files)

    splits = []
    for i in xrange(1, 4):
        train_list = [
            (vid, class_dict[cls[0]]) for cls in class_info_list for vid in cls[2] if cls[1] == i
        ]
        test_list = [
            (vid, class_dict[cls[0]]) for cls in class_info_list for vid in cls[3] if cls[1] == i
        ]
        splits.append((train_list, test_list))
    return splits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ucf101', choices=['ucf101', 'hmdb51'])
    parser.add_argument('--frame_path', type=str, default='./ucf101_frames',
                        help="root directory holding the frames")
    parser.add_argument('--out_list_path', type=str, default='./settings')

    parser.add_argument('--rgb_prefix', type=str, default='img_',
                        help="prefix of RGB frames")
    parser.add_argument('--flow_x_prefix', type=str, default='flow_x',
                        help="prefix of x direction flow images")
    parser.add_argument('--flow_y_prefix', type=str, default='flow_y',
                        help="prefix of y direction flow images", )

    parser.add_argument('--num_split', type=int, default=3,
                        help="number of split building file list")
    parser.add_argument('--shuffle', action='store_true', default=False)

    args = parser.parse_args()

    dataset = args.dataset
    frame_path = args.frame_path
    rgb_p = args.rgb_prefix
    flow_x_p = args.flow_x_prefix
    flow_y_p = args.flow_y_prefix
    num_split = args.num_split
    out_path = args.out_list_path
    shuffle = args.shuffle

    out_path = os.path.join(out_path,dataset)
    if not os.path.isdir(out_path):
        print("creating folder: "+out_path)
        os.makedirs(out_path)

    # operation
    print('processing dataset {}'.format(dataset))
    if dataset=='ucf101':
        split_tp = parse_ucf101_splits()
    else:
        split_tp = parse_hmdb51_splits()
    f_info = parse_directory(frame_path, rgb_p, flow_x_p, flow_y_p)

    print('writing list files for training/testing')
    for i in xrange(max(num_split, len(split_tp))):
        lists = build_split_list(split_tp, f_info, i, shuffle)
        open(os.path.join(out_path, 'train_rgb_split{}.txt'.format(i + 1)), 'w').writelines(lists[0][0])
        open(os.path.join(out_path, 'val_rgb_split{}.txt'.format(i + 1)), 'w').writelines(lists[0][1])
        open(os.path.join(out_path, 'train_flow_split{}.txt'.format(i + 1)), 'w').writelines(lists[1][0])
        open(os.path.join(out_path, 'val_flow_split{}.txt'.format(i + 1)), 'w').writelines(lists[1][1])

