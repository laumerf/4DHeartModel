import argparse
from pathlib import Path
from shutil import copy as cp


def add_zeros(string, n_zeros=3):
    new_string = ''
    num = ''
    for i, c in enumerate(string):
        if c.isdigit():
            if string[i+1].isdigit():
                num += c
            else:
                num += c
                new_string += num.zfill(n_zeros)
                num = ''
        else:
            new_string += c
    return new_string


def rename_all(root_dir, results_dir, pref):
    file_names = [f.name for f in Path(root_dir).iterdir()
                  if (root_dir/f).is_file() and f.name.startswith(pref)]
    renamed_names = [add_zeros(f) for f in file_names]

    files = [root_dir / f for f in file_names]
    renamed_files = [results_dir / f for f in renamed_names]

    [cp(str(f1), str(f2)) for f1, f2 in zip(files, renamed_files)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir',
                        type=lambda p: Path(p).absolute(),
                        help='Path to dir where reconstructed vtks are saved')
    parser.add_argument('--results_dir',
                        type=lambda p: Path(p).absolute(),
                        help='Path to dir where renamed results will be saved')
    parser.add_argument('--pref',
                        type=str,
                        help='Prefix of the files to be renamed')
    args = parser.parse_args()
    rename_all(args.root_dir, args.results_dir, args.pref)


if __name__ == "__main__":
    main()