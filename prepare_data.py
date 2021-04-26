from pathlib import Path
from data.data_pipe import load_bin, load_mx_rec
import args

if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='for face verification')
    # parser.add_argument("-r", "--rec_path", help="mxnet record file path",default='faces_emore', type=str)
    # args = parser.parse_args()
    # conf = get_config()
    # rec_path = conf.data_path/args.rec_path
    # load_mx_rec(rec_path)
    
    bin_files = ['agedb_30']
    rec_path = Path(args.eval_dataset)
    for i in range(len(bin_files)):
        load_bin(rec_path/(bin_files[i]+'.bin'), rec_path/bin_files[i], args.test_transform)