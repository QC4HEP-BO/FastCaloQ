import os
from argparse import ArgumentParser
from glob import glob
import pandas as pd
from pdb import set_trace

def completion_check(ifile):
    csv_file = f'{ifile}/evaluate/chi2.csv'
    df = pd.read_csv(f'{csv_file}')
    if df.shape[0] < 1001:
        print('\033[92m[INFO] Evaluation\033[0m', f'{csv_file}', df.shape[0])
    return df.shape[0]

def main(args):
    files = glob(f'{args.input}*/**/*eta*[!p][!d][!f]/', recursive=True)
    exclude = ['slope_study']
    files = [i for i in files for j in exclude if j not in i]

    def get_info(path):
        task = path.split('/')[-3].split('_')
        job = path.split('/')[-2].split('_')
        folder =  path.split('/')[-4]
        if len(task) < 2 or len(job) < 4:
            print('\033[91m[ERROR] job not found in\033[0m', f'{path}')
        return tuple([task[0], task[1], job[0], job[2], job[3], folder])

    pid = {
        'pions': 211,
        'photons': 22,
        'electrons': 11,
        'protons': 2212,
    }

    results = {'particle': [], 'eta': [], 'model': [], 'hp': [], 'chi2': [], 'iter': [], 'total': [], 'folder': []}

    for ifile in files:
        model, hp, particle, eta_min, eta_max, folder = get_info(ifile)
        csv_file = f'{ifile}/selected/chi2.csv'
        if not os.path.exists(csv_file):
            print('\033[92m[INFO] Unfinished job\033[0m', f'{csv_file}')
            continue
        tot_iter = completion_check(ifile)
        df = pd.read_csv(f'{csv_file}')
        chi2 = float(df['All'])
        iteration = int(df['ckpt'])

        results['particle'].append(particle)
        results['eta'].append(int(eta_min))
        results['model'].append(model)
        results['hp'].append(hp)
        results['chi2'].append(chi2)
        results['iter'].append(iteration)
        results['total'].append(tot_iter)
        results['folder'].append(folder)
    dfs = pd.DataFrame.from_dict(results).sort_values(by=['particle', 'eta', 'model', 'hp'])
    for particle, df in dfs.groupby(by=['particle']):
        out_folder = args.input+'/' if os.path.isdir(args.input) else (args.input.rstrip() + '_')
        df.to_csv(f'{out_folder}results_{particle}.csv', index=False)
        print('\033[92m[INFO] Save to\033[0m', f'{out_folder}results_{particle}.csv')
        os.system(f'tablign {out_folder}results_{particle}.csv')
        print()

if __name__ == '__main__':

    """Get arguments from command line."""
    parser = ArgumentParser(description="\033[92mConfig for training.\033[0m")
    parser.add_argument('-i', '--input', type=str, required=False, default='../output/dataset1/v1/', help='Training h5 file name (default: %(default)s)')

    args = parser.parse_args()
    main(args)
