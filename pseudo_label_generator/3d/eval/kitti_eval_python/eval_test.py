from evaluate import evaluate

det_path = "/path/to/Data_object_det/res_regnety"  # (redacted)
gt_path = "/path/to/data/kitti/training/label_2"  # (redacted)
gt_split_file = "/path/to/val.txt"  # (redacted) from https://xiaozhichen.github.io/files/mv3d/imagesets.tar.gz

for i in range(100):
    evaluate(gt_path,det_path,gt_split_file, score_thresh=i/100., current_class=[0])