from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("video", type=int)
video = parser.parse_args().video
if video == 1:
    VIDEO = "1_train-val_1min_aalesund_from_start"
elif video == 2:
    VIDEO = "2_train-val_1min_after_goal"
elif video == 3:
    VIDEO = "3_test_1min_hamkam_from_start"
elif video == 4:
    VIDEO = "4_annotate_1min_bodo_start"