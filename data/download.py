"""
DELRec/data/download.py
=======================
Amazon 2018 / 2023 数据集下载 & 解压工具。
可独立运行：
    python -m DELRec.data.download --year 2018 --category Movies_and_TV
    python -m DELRec.data.download --year 2023 --category Movies_and_TV
"""

import os
import gzip
import shutil
import requests
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# 下载基地址
# ─────────────────────────────────────────────────────────────────────────────
BASE_2018_REVIEW = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/categoryFiles/"
BASE_2018_META   = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_v2/metaFiles2/"

BASE_2023_REVIEW = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/"
BASE_2023_META   = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/meta_categories/"


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def download_file(url: str, out_path: str, timeout: int = 60) -> None:
    """
    下载单个文件到 out_path。
    - 若目标文件已存在，直接跳过。
    - 使用 .part 临时文件，下载完成后原子替换，避免残留不完整文件。
    """
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    if os.path.exists(out_path):
        print(f"[√] 已存在，跳过: {out_path}")
        return

    tmp_path = out_path + ".part"
    print(f"[↓] 下载中: {url}")
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"下载失败 {url}: {e}") from e

    total = int(r.headers.get("content-length", 0) or 0)
    with open(tmp_path, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=os.path.basename(out_path)
    ) as pbar:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    os.replace(tmp_path, out_path)
    print(f"[√] 下载完成: {out_path}")


def is_valid_gzip(gz_path: str) -> bool:
    """校验 gzip 文件完整性。"""
    try:
        with gzip.open(gz_path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except Exception:
        return False


def unzip_gz(gz_path: str) -> str:
    """
    解压 .gz 文件，返回解压后路径。
    - 若已解压，直接返回。
    - 若 gzip 损坏，删除并抛出异常。
    """
    out_path = gz_path[:-3]

    if os.path.exists(out_path):
        print(f"[√] 已解压，跳过: {out_path}")
        return out_path

    if not os.path.exists(gz_path):
        raise FileNotFoundError(f"找不到 gz 文件: {gz_path}")

    if not is_valid_gzip(gz_path):
        print(f"[!] gzip 文件损坏，已删除: {gz_path}")
        os.remove(gz_path)
        raise RuntimeError(f"gzip 文件损坏: {gz_path}")

    print(f"[↪] 解压中: {gz_path}")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

    print(f"[√] 解压完成: {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Amazon 2018 / 2023 下载入口
# ─────────────────────────────────────────────────────────────────────────────

def ensure_amazon_2018(category: str, root: str = "data/amazon_2018"):
    """
    确保 Amazon 2018 指定类别数据已下载并解压。

    Parameters
    ----------
    category : str  例如 "Movies_and_TV"、"Sports_and_Outdoors"
    root     : str  数据根目录

    Returns
    -------
    review_path, meta_path : (str, str)
    """
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    review_path = os.path.join(raw_dir, f"{category}.json")
    meta_path   = os.path.join(raw_dir, f"meta_{category}.json")

    if os.path.exists(review_path) and os.path.exists(meta_path):
        print(f"[√] Amazon 2018 [{category}] 已准备好。")
        return review_path, meta_path

    review_gz = review_path + ".gz"
    meta_gz   = meta_path + ".gz"

    download_file(BASE_2018_REVIEW + os.path.basename(review_gz), review_gz)
    download_file(BASE_2018_META   + os.path.basename(meta_gz),   meta_gz)

    review_path = unzip_gz(review_gz)
    meta_path   = unzip_gz(meta_gz)

    return review_path, meta_path


def ensure_amazon_2023(category: str, root: str = "data/amazon_2023"):
    """
    确保 Amazon 2023 指定类别数据已下载并解压。

    Parameters
    ----------
    category : str  例如 "Movies_and_TV"
    root     : str  数据根目录

    Returns
    -------
    review_path, meta_path : (str, str)
    """
    raw_dir = os.path.join(root, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    review_path = os.path.join(raw_dir, f"{category}.jsonl")
    meta_path   = os.path.join(raw_dir, f"meta_{category}.jsonl")

    if os.path.exists(review_path) and os.path.exists(meta_path):
        print(f"[√] Amazon 2023 [{category}] 已准备好。")
        return review_path, meta_path

    review_gz = review_path + ".gz"
    meta_gz   = meta_path + ".gz"

    download_file(BASE_2023_REVIEW + os.path.basename(review_gz), review_gz)
    download_file(BASE_2023_META   + os.path.basename(meta_gz),   meta_gz)

    review_path = unzip_gz(review_gz)
    meta_path   = unzip_gz(meta_gz)

    return review_path, meta_path


def ensure_amazon_dataset(year: int, category: str):
    """
    统一入口。

    Parameters
    ----------
    year     : 2018 或 2023
    category : 类别名称字符串

    Returns
    -------
    review_path, meta_path : (str, str)
    """
    if year == 2018:
        return ensure_amazon_2018(category)
    elif year == 2023:
        return ensure_amazon_2023(category)
    else:
        raise ValueError(f"不支持的 Amazon 年份: {year}，请使用 2018 或 2023")


# ─────────────────────────────────────────────────────────────────────────────
# CLI 入口
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="下载 Amazon 数据集")
    parser.add_argument("--year",     type=int, required=True, help="2018 或 2023")
    parser.add_argument("--category", type=str, required=True,
                        help="类别名称，例如 Movies_and_TV")
    args = parser.parse_args()

    paths = ensure_amazon_dataset(args.year, args.category)
    print("完成:", paths)
