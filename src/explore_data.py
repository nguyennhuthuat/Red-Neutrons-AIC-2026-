import os 
import json 
import numpy as np 
from pathlib import Path 
import pandas as pd 
import subprocess 
    

def download_dataset(dataset_slug, target_dir = "data/hcmc2023", unzip = True):
    target = Path(target_dir)
    if target.exists() and any(target.iterdir()): 
        print(f"Dataset đã có ở {target_dir}, không cần tải")
        return 
    os.makedirs(target_dir, exist_ok=True)
    cmd = f"kaggle datasets download -d {dataset_slug} -p {target_dir}"
    if unzip: 
        cmd += " --unzip"
    os.system(cmd)

    print("DONE!")

def explore_directory_tree(root_dir, max_depth = 3, max_items = 6): # In ra cấu trúc cây của thư mục
    root = Path(root_dir)
    if not root.exists(): 
        print(f"Không tìm thấy {root}")
        return 
    
    def walk(path, depth = 0): 
        if depth > max_depth: 
            return 
        items = sorted(path.iterdir())
        dirs = [x for x in items if x.is_dir()]
        files = [x for x in items if x.is_file()]

        for d in dirs[:max_items]: 
            print("  "*depth + f"{d.name}/")
            walk(d, depth + 1)
        
        if len(dirs) > max_items: 
            print("  " * depth + f" ... và {len(dirs) - max_items} thư mục nữa")

        if files: 
            ext_count = {}
            for f in files: 
                ext_count[f.suffix] = ext_count.get(f.suffix, 0) + 1 
            for ext, count in ext_count.items():
                sample = next(f for f in files if f.suffix == ext)
                print("  " *depth + f"{count} file {ext or '(no ext)'} - vd: {sample.name}")

    walk(root)

def inspect_npy_files(root_dir, max_files = 3): 
    root = Path(root_dir)
    npy_files = list(root.rglob("*.npy"))
    print(f"File .npy tìm thấy {len(npy_files)}")

    if not npy_files: 
        print("Không thấy .npy - maybe feature lưu dạng khác")
        return 
    
    for f in npy_files[:max_files]: 
        try: 
            arr = np.load(f, allow_pickle=True)
            print(f"\n {f.relative_to(root)}")
            print(f" Shape: {arr.shape}")
            print(f" Dtype: {arr.dtype}")
            if arr.ndim == 2: 
                print(f"-> {arr.shape[0]} vectors, mỗi vector {arr.shape[1]} chiều")
                print(f"-> CLIP dim = {arr.shape[1]}")

        except Exception as e: 
            print(f"Không đọc được {f.name}: {e}")

def inspect_other_feature(root_dir): 
    root = Path(root_dir)
    exts = [".pt", ".pth", ".pkl", ".h5", ".hdf5", ".bin"]
    for ext in exts: 
        files = list(root.rglob(f"*{ext}"))
        if files: 
            print(f"Tìm thấy {len(files)} file {ext} - vd: {files[0].relative_to(root)}")

def inspect_metadata(root_dir): 
    root = Path(root_dir)
    csvs = list(root.rglob("*.csv"))
    print(f"\nCSV: {len(csvs)} file")
    if csvs: 
        df = pd.read_csv(csvs[0])
        print(f" vd {csvs[0].name} -> cột {list(df.columns)}")
        print(df.head(3))

    jsons = list(root.rglob("*json"))
    print(f"\nJSON: {len(jsons)} file")
    if jsons: 
        with open(jsons[0], encoding="utf-8") as fp: 
            data = json.load(fp)
        keys = list(data.keys()) if isinstance(data, dict) else "(list)"
        print(f" vd {jsons[0].name} -> keys: {keys}")

def inspect_keyframes(root_dir): 
    root = Path(root_dir)
    imgs = list(root.rglob("*.jpg")) + list(root.rglob("*.png"))
    print(f"\nKeyframe ảnh: {len(imgs)} file")
    if imgs: 
        print(f" vd: {imgs[0].relative_to(root)}")

def explore_all(root_dir): 

    explore_directory_tree(root_dir)
    inspect_npy_files(root_dir)
    inspect_other_feature(root_dir)
    inspect_metadata(root_dir) 
    inspect_keyframes(root_dir)

if __name__ == "__main__": 

    DATA_DIR = "data/hcmc2023"
    SLUG_DATASET = "mkimwp/hcmc-ai-challenge-2023"

    download_dataset(SLUG_DATASET, DATA_DIR)

    explore_all(DATA_DIR)