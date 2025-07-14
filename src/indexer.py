import os
import argparse
from src.storage import init_db
from src.metadata import extract_metadata

def run_indexing(root_dir, db_path):
    print(f"[indexer.py] Indexing images from: {root_dir}")
    conn = init_db(db_path)
    sql = ('INSERT OR REPLACE INTO images(file_path, brightness, contrast, '
           'sharpness, entropy, height, width) VALUES (?, ?, ?, ?, ?, ?, ?)')
    count = 0
    for class_name in os.listdir(root_dir):
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            path = os.path.join(class_dir, fname)
            meta = extract_metadata(path)
            params = (
                path,
                meta['brightness'],
                meta['contrast'],
                meta['sharpness'],
                meta['entropy'],
                meta['height'],
                meta['width']
            )
            try:
                conn.execute(sql, params)
                count += 1
                if count % 1000 == 0:
                    print(f"[indexer.py] Indexed {count} images so far...")
            except Exception as e:
                print(f"Error indexing {path}: {e}")
    print(f"[indexer.py] Indexing complete. Total images indexed: {count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Index CIFAR-10 metadata into DuckDB.')
    parser.add_argument('--root_dir', default='data/raw/train', help='Root image folder for indexing')
    parser.add_argument('--db_path', default='cifar_meta.db', help='Output DuckDB file path')
    args = parser.parse_args()
    run_indexing(args.root_dir, args.db_path)