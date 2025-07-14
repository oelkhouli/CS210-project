import argparse
from src.storage import init_db

def get_file_list(min_bright=None,
                  max_entropy=None,
                  min_contrast=None,
                  min_sharpness=None,
                  limit=20,
                  db_path='cifar_meta.db'):
    """
    Retrieve a list of image file paths filtered by metadata.
    Args:
        min_bright (float): minimum brightness
        max_entropy (float): maximum entropy
        min_contrast (float): minimum contrast
        min_sharpness (float): minimum sharpness
        limit (int): max number of results (None for no limit)
        db_path (str): path to DuckDB database file
    Returns:
        List[str]: matching image file paths
    """
    conn = init_db(db_path)
    clauses = []
    if min_bright is not None:
        clauses.append(f"brightness >= {min_bright}")
    if max_entropy is not None:
        clauses.append(f"entropy <= {max_entropy}")
    if min_contrast is not None:
        clauses.append(f"contrast >= {min_contrast}")
    if min_sharpness is not None:
        clauses.append(f"sharpness >= {min_sharpness}")
    where = ' AND '.join(clauses) if clauses else '1=1'

    query = f"SELECT file_path FROM images WHERE {where}"
    if limit is not None:
        query += f" LIMIT {limit}"
    query += ';'

    rows = conn.execute(query).fetchall()
    return [r[0] for r in rows]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Query metadata-filtered image paths")
    parser.add_argument('--min-bright',   type=float, help='Minimum brightness filter')
    parser.add_argument('--max-entropy',  type=float, help='Maximum entropy filter')
    parser.add_argument('--min-contrast', type=float, help='Minimum contrast filter')
    parser.add_argument('--min-sharpness',type=float, help='Minimum sharpness filter')
    parser.add_argument('--limit',        type=int,   help='Limit number of results')
    parser.add_argument('--db_path',      type=str, default='cifar_meta.db', help='Path to DuckDB database')
    args = parser.parse_args()

    results = get_file_list(
        min_bright=args.min_bright,
        max_entropy=args.max_entropy,
        min_contrast=args.min_contrast,
        min_sharpness=args.min_sharpness,
        limit=args.limit,
        db_path=args.db_path
    )
    for path in results:
        print(path)
