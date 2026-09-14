# -*- coding: utf-8 -*-
"""
download_mp_data.py  （并发逐 id 版 v4）
从 Materials Project 下载 晶体结构 + 3 个性质，生成 CGCNN 多任务数据集。

为什么是"并发逐 id"而不是"批量"？
—— v3 实验证明：把多个 id 用逗号拼成一个 material_ids 参数请求，
   summary 端点会返回空（0 个结果）。只有"一个请求一个 id"才可靠
   （这正是 pymatgen 自带 get_structure_by_material_id 的用法）。
   所以本版：每个 id 发 1 次请求（structure + 3 性质一次拿回），
   用多线程并发把速度提上去。

用法：
    $env:MP_API_KEY = "你的key"
    python download_mp_data.py --limit 2000            # 第 1 步：测速
    python download_mp_data.py --limit 50000           # 第 2 步：全量（46744 个）
    python download_mp_data.py --limit 50000 --workers 16   # 网络好可加并发
"""
import argparse
import os
import sys
import time
import shutil
from concurrent.futures import ThreadPoolExecutor

from pymatgen.ext.matproj import MPRester

# ---------- 设置 ----------
IDS_FILE = 'data/material-data/mp-ids-46744.csv'
OUT_DIR = 'data/mt-large'
PROPS = ['formation_energy_per_atom', 'band_gap', 'efermi']
ATOM_INIT_SRC = 'data/sample-regression/atom_init.json'
PRINT_EVERY = 500     # 每处理多少个 id 打印一次进度/写一次盘
# -----------------------


def save_state(csv_path, rows_dict, ids):
    """把进度写进 id_prop.csv（无表头）。csv 就是断点续传的依据。"""
    with open(csv_path, 'w') as f:
        for mid in ids:
            if mid in rows_dict:
                f.write(','.join([mid] + rows_dict[mid]) + '\n')


def make_fetcher(api_key, fields):
    """单个 id 的下载任务：1 次请求拿 structure + 3 性质。返回 (mid, doc, err)。"""
    def fetch_one(mid):
        for attempt in range(3):
            try:
                with MPRester(api_key) as mpr:
                    url = 'materials/summary/?_fields=' + fields
                    docs = mpr.request(url, payload={'material_ids': mid})
                if not docs:
                    return mid, None, None       # 空 = 该 id 已失效
                return mid, docs[0], None
            except Exception as e:
                if attempt == 2:
                    return mid, None, 'ERR_' + type(e).__name__
                time.sleep(2 * (attempt + 1))    # 退避重试
    return fetch_one


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=0,
                        help='下载前多少个 mp-id，0=全部（默认 0）')
    parser.add_argument('--workers', type=int, default=8,
                        help='并发线程数（默认 8，被限流就调小，网络好可调大）')
    args = parser.parse_args()

    api_key = os.environ.get('MP_API_KEY')
    if not api_key:
        sys.exit('没有找到 MP_API_KEY。请先运行:  $env:MP_API_KEY = "你的key"')

    # 1. 读 id 清单
    with open(IDS_FILE) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    ids = all_ids if args.limit <= 0 else all_ids[:args.limit]
    print('任务：共 {} 个 mp-id（文件共 {} 个，并发 {}）'.format(
        len(ids), len(all_ids), args.workers))

    # 2. 准备目录
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, 'id_prop.csv')
    shutil.copyfile(ATOM_INIT_SRC, os.path.join(OUT_DIR, 'atom_init.json'))

    # 3. 载入已有进度
    rows_dict = {}
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) == 1 + len(PROPS):
                    rows_dict[parts[0]] = parts[1:]
    todo = [mid for mid in ids if mid not in rows_dict]
    print('已有 {} 个，本次需下载 {} 个'.format(len(rows_dict), len(todo)))
    if not todo:
        print('全部完成，无需下载。')
        return

    # 4. 并发逐 id 下载
    fields = 'structure,' + ','.join(PROPS)
    fetch_one = make_fetcher(api_key, fields)

    start = time.time()
    new_count, dead_count, err_count = 0, 0, 0
    shown_err = 0
    processed = 0

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        for mid, doc, err in ex.map(fetch_one, todo, chunksize=32):
            processed += 1
            if err is not None:
                err_count += 1
                if shown_err < 5:
                    print('  !! {} {}'.format(mid, err))
                    shown_err += 1
            elif doc is None:
                dead_count += 1
            else:
                try:
                    doc['structure'].to(filename=os.path.join(OUT_DIR, mid + '.cif'))
                    values = ['' if doc.get(p) is None else str(doc.get(p))
                              for p in PROPS]
                    rows_dict[mid] = values
                    new_count += 1
                except Exception as e:
                    err_count += 1
                    if shown_err < 5:
                        print('  !! 写文件失败 {}: {}'.format(mid, type(e).__name__))
                        shown_err += 1

            # 5. 定期写盘 + 打印速度/ETA
            if processed % PRINT_EVERY == 0 or processed == len(todo):
                save_state(csv_path, rows_dict, ids)
                elapsed_min = (time.time() - start) / 60.0
                rate = processed / elapsed_min if elapsed_min > 0 else 0
                remaining = len(todo) - processed
                eta_h = remaining / rate / 60.0 if rate > 0 else 0
                print('  进度 {}/{} | 本次已下 {} | 累计有效 {} | 用时 {:.1f} 分 | '
                      '速度 ~{:.0f} 个/分 | 预计还需 ~{:.1f} 小时'.format(
                          processed, len(todo), new_count, len(rows_dict),
                          elapsed_min, rate, eta_h))

    # 6. 最终写盘 + 报告
    save_state(csv_path, rows_dict, ids)
    n_cif = len([f for f in os.listdir(OUT_DIR) if f.endswith('.cif')])
    print()
    print('==== 完成 ====')
    print('  本次新下载: {} 个'.format(new_count))
    print('  累计有效: {} 个（id_prop.csv 行数）'.format(len(rows_dict)))
    print('  CIF 文件数: {}'.format(n_cif))
    print('  本次失效: {} 个'.format(dead_count))
    print('  失败/出错: {} 个（重跑同命令可补）'.format(err_count))
    print('  总用时: {:.1f} 分钟'.format((time.time() - start) / 60.0))
    print('  输出目录: {}/'.format(OUT_DIR))


if __name__ == '__main__':
    main()
