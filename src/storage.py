import duckdb

def init_db(db_path='cifar_meta.db'):
    # Initialize DuckDB connection and create images table
    conn = duckdb.connect(db_path)
    conn.execute('''
    CREATE TABLE IF NOT EXISTS images (
        file_path VARCHAR PRIMARY KEY,
        brightness DOUBLE,
        contrast DOUBLE,
        sharpness DOUBLE,
        entropy DOUBLE,
        height INTEGER,
        width INTEGER
    );
    ''')
    return conn