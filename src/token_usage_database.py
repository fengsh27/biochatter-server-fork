
import sqlite3
from sqlite3 import Connection
import os
from time import strftime
from typing import Optional
import logging

from src.constants import (
    DATA_DIR
)

logger = logging.getLogger(__name__)

create_table_query = """
CREATE TABLE IF NOT EXISTS AzureOpenAITokenUsage (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT NOT NULL,
    hour TEXT NOT NULL,
    user TEXT NOT NULL,
    model TEXT not NULL,
    completion_tokens INTEGER,
    prompt_tokens INTEGER,
    total_tokens INTEGER,
    UNIQUE (user, model, date, hour)
);
"""

def _ensure_azure_token_usage_tables(conn: Optional[Connection]=None):
    if conn is None:
        return
    try:
        cursor = conn.cursor()
        cursor.execute(create_table_query)
        conn.commit()
    except Exception as e:
        logger.error(e)
    pass

def _connect_to_db():
    db_path = os.environ.get(DATA_DIR, "./data")
    db_path = os.path.join(db_path, "token_usage.db")
    if not os.path.exists(db_path):
        try:
            with open(db_path, "w"):
                pass
        except Exception as e:
            logger.error(e)
            return None
    return sqlite3.connect(db_path)

upsert_usage_query = """
insert into AzureOpenAITokenUsage(user, model, date, hour, completion_tokens, prompt_tokens, total_tokens) 
values (?,?,?,?,?,?,?) 
on conflict(user, model, date, hour)
do update set completion_tokens=completion_tokens+excluded.completion_tokens,
prompt_tokens=prompt_tokens+excluded.prompt_tokens,
total_tokens=total_tokens+excluded.total_tokens;
"""
def _update_token_usage(conn: Connection, user: str, model: str, token_usage: dict):
    date=strftime("%Y-%m-%d")
    hour=strftime("%H")
    cursor = conn.cursor()        
    cursor.execute(upsert_usage_query,
                    (user, model, date, hour,
                    token_usage["completion_tokens"],
                    token_usage["prompt_tokens"],
                    token_usage["total_tokens"]))
    conn.commit()

def _get_token_usage(conn: Connection, user: str, model: str):
    pass

def update_token_usage(user: str, model: str, token_usage: dict):
    conn = _connect_to_db()
    if conn is None:
        return
    _ensure_azure_token_usage_tables(conn)
    try:
        _update_token_usage(conn, user, model, token_usage)
    except Exception as e:
        logger.error(e)
    finally:
        conn.close()

select_usage_query = """
select completion_tokens, prompt_tokens, total_tokens from AzureOpenAITokenUsage
where user=? and model=? and date=?;
"""
def get_token_usage(user: str, model: str):
    date = strftime("%Y-%m-%d")
    conn = _connect_to_db()
    if conn is None:
        return None
    try:
        cursor = conn.cursor()
        cursor.execute(select_usage_query, (user, model, date))
        res = cursor.fetchone()
        cursor.close()
        return res
    except Exception as e:
        logger.error(e)
        return None
    finally:
        conn.close()
    
