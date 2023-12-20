"""Helper module for EDA notebook to perform
data cleaning and preprocessing"""
import pandas as pd
from time import time
from collections import Counter
import duckdb
import sqlite3


def check_nulls_sqlite(table_name: str, cursor: sqlite3.Cursor) -> None:
    """Check null values from all the columns with SQLite"""
    query: str = "SELECT "
    for col in cursor.execute(
        f"SELECT name FROM pragma_table_info('{table_name}')"
    ).fetchall():
        query += f"COUNT(*) - COUNT({col[0]}) AS {col[0]}, "
    query = query[:-2]  # Remove last comma.
    query += f" FROM {table_name}"
    rows: list[str] = cursor.execute(query).fetchall()
    names: list[str] = [description[0] for description in cursor.description]
    print("Number of null values:")
    for row in rows:
        for name, val in zip(names, row):
            print(name, val)


def check_dublicates_sqlite(table_name, cursor) -> None:
    """Check null values from all the columns with SQLite"""
    cols: list[str] = [
        col[0]
        for col in cursor.execute(
            f"SELECT name FROM pragma_table_info('{table_name}')"
        ).fetchall()
    ]
    query: str = "SELECT COUNT(*)"
    query += f" FROM {table_name}"
    query += f" GROUP BY {', '.join(cols)} HAVING COUNT(*) > 1"
    rows: list[str] = cursor.execute(query).fetchone()
    print("Number of dublicate rows:")
    if rows:
        print(rows)
    else:
        print(0)


def drop_rows_with_nulls_sqlite(table_name, new_table_name, cursor):
    """Create a new temporary table without rows with NULL values"""
    query: str = f"CREATE TEMPORARY TABLE {new_table_name} AS "
    query += f"SELECT * FROM {table_name} WHERE "
    for col in cursor.execute(
        f"SELECT name FROM pragma_table_info('{table_name}')"
    ).fetchall():
        query += f"{col[0]} IS NOT NULL AND "
    query = query[:-4]  # Remove last AND.
    cursor.execute(query)


def find_team_overall_scores(match_table: str, player_attributes: str
                             ) -> pd.DataFrame:
    """Combine player attributes for each match and returns it as dataframe."""
    print("Combining data.")
    start = time()
    home_players: list[str] = ["home_player_" + str(x) for x in range(1, 12)]
    away_players: list[str] = ["away_player_" + str(x) for x in range(1, 12)]

    query_select: str = """ SELECT m.id, """
    query_joins: str = f""" FROM {match_table} m """
    for i, player in enumerate(home_players + away_players):
        query_select += f" tb_{i}.overall_rating AS {player}_rating, "
        query_joins += f""" LEFT JOIN {player_attributes} tb_{i} 
                    ON tb_{i}.player_api_id = m.{player}
                    AND tb_{i}.date = (
                    SELECT MAX(date) 
                    FROM {player_attributes} 
                    WHERE player_api_id = m.{player} AND date <= m.date
                ) """

    query_select = query_select[:-2]  # Remove last comma.
    query = query_select + " " + query_joins
    df = duckdb.query(query).df()
    end = time()
    print(f"Data collected in {(end - start) / 60:.1f} minutes")
    return df


def find_team_formation(match) -> tuple[str, str]:
    """Find formation of teams"""
    # The position of goal keeper is the same in every formation 
    # so it is not included.
    coordinates_columns: list[str] = [
                "player_Y" + str(X) for X in range(2, 12)
                ]
    h_coordinates: list[int] = []
    a_coordinates: list[int] = []
    for position in coordinates_columns:
        h_coordinates.append(match["home_" + position])
        a_coordinates.append(match["away_" + position])

    home_format: list[int] = list(Counter(sorted(h_coordinates)).values())
    away_format: list[int] = list(Counter(sorted(a_coordinates)).values())
    home_formation_str: str = "".join(str(c) for c in home_format)
    away_formation_str: str = "".join(str(c) for c in away_format)
    return home_formation_str, away_formation_str


def calculate_team_ratings_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the average overall rating for each team in a match"""
    home_ratings_columns: list[str] = [
        f"home_player_{i}_rating" for i in range(1, 12)
        ]
    away_ratings_columns: list[str] = [
        f"away_player_{i}_rating" for i in range(1, 12)
        ]

    df["home_avg_rating"] = df[home_ratings_columns].mean(axis=1)
    df["home_std_rating"] = df[home_ratings_columns].std(axis=1)

    df["away_avg_rating"] = df[away_ratings_columns].mean(axis=1)
    df["away_std_rating"] = df[away_ratings_columns].std(axis=1)

    df.drop(columns=home_ratings_columns + away_ratings_columns, inplace=True)
    return df

