import argparse


def motMetricsEnhancedCalculator(gtSource, tSource, only_calculate_class=None):
    # import required packages
    import motmetrics as mm
    import numpy as np
    
    # load ground truth
    gt = np.loadtxt(gtSource, delimiter=',')

    # load tracking output
    t = np.loadtxt(tSource, delimiter=',')

    # Create an accumulator that will be updated during each frame
    acc = mm.MOTAccumulator(auto_id=True)

    # Max frame number maybe different for gt and t files
    for frame in range(int(gt[:,0].max())):
        frame += 1 # detection and frame numbers begin at 1

        # select id, x, y, width, height for current frame
        # required format for distance calculation is X, Y, Width, Height \
        # We already have this format
        if only_calculate_class is not None:
            gt_dets = gt[(gt[:,0]==frame) & (gt[:,7]==only_calculate_class),1:6] # select all detections in gt
            t_dets = t[(t[:,0]==frame) & (t[:,7]==only_calculate_class),1:6] # select all detections in t
        else:
            gt_dets = gt[gt[:,0]==frame,1:6] # select all detections in gt
            t_dets = t[t[:,0]==frame,1:6]
        C = mm.distances.iou_matrix(gt_dets[:,1:], t_dets[:,1:], \
                                    max_iou=0.5) # format: gt, t

        # Call update once for per frame.
        # format: gt object ids, t object ids, distance
        acc.update(gt_dets[:,0].astype('int').tolist(), \
                   t_dets[:,0].astype('int').tolist(), C)

    mh = mm.metrics.create()

    summary = mh.compute(acc, metrics=['num_frames', 'idf1', 'idp', 'idr', \
                                        'recall', 'precision', 'num_objects', \
                                        'mostly_tracked', 'partially_tracked', \
                                        'mostly_lost', 'num_false_positives', \
                                        'num_misses', 'num_switches', \
                                        'num_fragmentations', 'mota', 'motp' \
                                        ], \
                          name='acc')

    strsummary = mm.io.render_summary(
        summary,
        #formatters={'mota' : '{:.2%}'.format},
        namemap={'idf1': 'IDF1', 'idp': 'IDP', 'idr': 'IDR', 'recall': 'Rcll', \
                'precision': 'Prcn', 'num_objects': 'GT', \
                'mostly_tracked' : 'MT', 'partially_tracked': 'PT', \
                'mostly_lost' : 'ML', 'num_false_positives': 'FP', \
                'num_misses': 'FN', 'num_switches' : 'IDsw', \
                'num_fragmentations' : 'FM', 'mota': 'MOTA', 'motp' : 'MOTP',  \
                }
    )
    print(strsummary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate MOT metrics')
    parser.add_argument('--gt', type=str, default="data/from_idun/3_test_1min_hamkam_from_start/gt/gt.txt", help='Ground truth file')
    parser.add_argument('--target', type=str, default="outputs/mot_result_3_test_1min_hamkam_from_start.txt", help='Tracking output file')
    parser.add_argument('--class-label', type=int, default=None)
    args = parser.parse_args()
    motMetricsEnhancedCalculator(args.gt, args.target, args.class_label)