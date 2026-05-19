# db config
from sqlalchemy import create_engine

DB_USER = "postgres"
DB_PASSWORD = "27agustus2005"
DB_HOST = "localhost"
DB_PORT = "5434"
DB_NAME = "keuangan_db"

DATABASE_URL = (
    f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(DATABASE_URL)
