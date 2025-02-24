import os
import psycopg2
import mariadb
from dotenv import load_dotenv

load_dotenv()

DB_TYPE = os.getenv("DB_TYPE", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432" if DB_TYPE == "postgres" else "3306")
DB_NAME = os.getenv("DB_NAME", "vector_store")
DB_USER = os.getenv("DB_USER", "admin")
DB_PASSWORD = os.getenv("DB_PASSWORD", "password")

def get_postgres_connection():
    return psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )

def get_mariadb_connection():
    return mariadb.connect(
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=int(DB_PORT),
        database=DB_NAME
    )

def get_connection():
    return get_postgres_connection() if DB_TYPE == "postgres" else get_mariadb_connection()

def create_table():
    query = """
    CREATE TABLE IF NOT EXISTS vector_store (
        id SERIAL PRIMARY KEY,
        description TEXT UNIQUE,
        generated_code TEXT,
        created_at TIMESTAMP DEFAULT NOW()
    );
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    conn.close()

def save_to_vector_store(description: str, generated_code: str):
    conn = get_connection()
    cursor = conn.cursor()
    query = """
    INSERT INTO vector_store (description, generated_code)
    VALUES (%s, %s)
    ON CONFLICT (description) DO UPDATE
    SET generated_code = EXCLUDED.generated_code;
    """
    cursor.execute(query, (description, generated_code))
    conn.commit()
    cursor.close()
    conn.close()

def check_vector_store(description: str):
    conn = get_connection()
    cursor = conn.cursor()
    query = "SELECT generated_code FROM vector_store WHERE description = %s;"
    cursor.execute(query, (description,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None

create_table()
