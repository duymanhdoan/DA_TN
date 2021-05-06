from pathlib import Path
from data.data_pipe import load_bin, load_mx_rec
import args

if __name__ == '__main__':
    rec_path = args.emore_folder
    load_mx_rec(rec_path)
    
    bin_files = [ 'lfw', 'cplfw' ,'cfp_fp' ,'cfp_ff' ,'calfw' ,'agedb_30','vgg2_fp']
    for i in range(len(bin_files)):
        print('load {}....\n'.format(bin_files[i]))
        load_bin(rec_path/(bin_files[i]+'.bin'), rec_path/bin_files[i], args.test_transform)