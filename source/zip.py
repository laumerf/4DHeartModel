import time
from zipfile import ZipFile
from pathlib import Path

def zip_exp_dir(exp_dir):

    print("Zipping experiment Files...")

    t1 = time.time()
  
    exp_dir = Path(exp_dir)

    files = []
    dirs = []
    for p in exp_dir.iterdir():
        if p.is_file():
            files.append(p)
        elif p.is_dir():
            dirs.append(p)
    
    for d in dirs:
        zip_path = exp_dir.absolute().parent / "zip" / d.name
        zip_path.mkdir(parents=True, exist_ok=True)
        zip_file = str(zip_path / exp_dir.name) + ".zip"
        # print(zip_file)
        with ZipFile(zip_file, 'w') as zipped:
            for f in d.glob('**/*'):
                name_in_zip = exp_dir.name + "/" + d.name + "/" + f.name
                zipped.write(f, name_in_zip)
    
    # zip files: config, scales, logs, ....
    zip_path = exp_dir.absolute().parent / "zip/files"
    zip_path.mkdir(parents=True, exist_ok=True)
    zip_file = str(zip_path / exp_dir.name) + ".zip"
    # print(zip_file)
    with ZipFile(zip_file, 'w') as zipped:
        for f in files:
            name_in_zip = exp_dir.name + "/" + f.name
            zipped.write(f, name_in_zip)

    t2 = time.time()

    print("Done zipping in {}s".format(t2-t1))
