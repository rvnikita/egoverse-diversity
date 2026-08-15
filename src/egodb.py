"""Connect to the EgoVerse episode registry (Postgres) and query it as a DataFrame.

Standalone on purpose: it does NOT import `egomimic`, so none of the training stack
(torch, mujoco-py, ros, projectaria-tools) needs to install. All it needs is
`~/.egoverse_env`, written by:

    AWS_PROFILE=egoverse bash external/EgoVerse/egomimic/utils/aws/setup_secret.sh

Usage:
    from egodb import episodes
    df = episodes()                      # whole table as a DataFrame
    df = episodes(where="task = 'fold_clothes'")
"""

from __future__ import annotations

import json
import os
import pathlib
import shlex

ENV_FILE = pathlib.Path.home() / ".egoverse_env"

# Columns of `TableRow` in egomimic/utils/aws/aws_sql.py, for reference:
#   episode_hash, operator, lab, task, embodiment, rig_name, num_frames,
#   task_description, scene, objects, zarr_processed_path, zarr_mp4_path,
#   zarr_processing_error, is_deleted, is_eval, eval_score, eval_success


def load_env(path: pathlib.Path = ENV_FILE) -> dict[str, str]:
    """Read ~/.egoverse_env. Values are shell-quoted by setup_secret.sh, so unquote."""
    if not path.exists():
        raise RuntimeError(
            f"{path} not found. Run:\n"
            "  AWS_PROFILE=egoverse bash external/EgoVerse/egomimic/utils/aws/setup_secret.sh"
        )
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, raw = line.partition("=")
        # printf %q output — shlex handles the quoting/escaping it produces.
        parts = shlex.split(raw)
        env[key] = parts[0] if parts else ""
    os.environ.update(env)
    return env


def db_config() -> dict:
    """Pull the Postgres connection details out of Secrets Manager."""
    import boto3

    load_env()
    arn = os.environ.get("SECRETS_ARN")
    if not arn:
        raise RuntimeError("SECRETS_ARN missing from ~/.egoverse_env")
    # setup_secret.sh is run under the egoverse profile; honour it here too unless
    # the caller has already picked one.
    session = boto3.Session(
        profile_name=os.environ.get("AWS_PROFILE", "egoverse"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-2"),
    )
    secret = session.client("secretsmanager").get_secret_value(SecretId=arn)
    return json.loads(secret["SecretString"])


def engine():
    from sqlalchemy import URL, create_engine

    cfg = db_config()
    url = URL.create(
        "postgresql+psycopg2",
        username=cfg.get("username", cfg.get("user")),
        password=cfg.get("password"),
        host=cfg.get("host"),
        port=int(cfg.get("port", 5432)),
        database=cfg.get("dbname", "appdb"),
    )
    return create_engine(url, connect_args={"connect_timeout": 20})


def tables() -> list[tuple[str, str]]:
    """(schema, table) pairs we're allowed to see — useful when the layout is unknown."""
    from sqlalchemy import text

    with engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT table_schema, table_name FROM information_schema.tables "
                "WHERE table_schema NOT IN ('pg_catalog','information_schema') "
                "ORDER BY 1,2"
            )
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def columns(table: str = "episodes", schema: str = "app") -> list[tuple[str, str]]:
    from sqlalchemy import text

    with engine().connect() as conn:
        rows = conn.execute(
            text(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_schema=:s AND table_name=:t ORDER BY ordinal_position"
            ),
            {"s": schema, "t": table},
        ).fetchall()
    return [(r[0], r[1]) for r in rows]


def episodes(where: str = "", limit: int | None = None,
             table: str = "app.episodes"):
    """The episode registry as a pandas DataFrame."""
    import pandas as pd

    sql = f"SELECT * FROM {table}"
    if where:
        sql += f" WHERE {where}"
    if limit:
        sql += f" LIMIT {int(limit)}"
    with engine().connect() as conn:
        return pd.read_sql(sql, conn)


if __name__ == "__main__":
    print("tables visible:")
    for s, t in tables():
        print(f"  {s}.{t}")
