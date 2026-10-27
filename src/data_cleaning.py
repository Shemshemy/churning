"""DuckDB view setup for the supplied iGaming transaction data."""

from pathlib import Path
from typing import Any


RAW_TABLE_FILES = {
    "Demographics": "SecondSeshData1Demographics.csv",
    "CashGames": "SecondSeshData2CashGames.csv",
    "Tournaments": "SecondSeshData3Tournaments.csv",
    "Deposits": "SecondSeshData4Deposits.csv",
    "Withdrawals": "SecondSeshData5Withdrawals.csv",
}

CLEANING_SQL = """
CREATE OR REPLACE VIEW Clean_Demographics AS
SELECT CAST(UserID AS BIGINT) AS UserID,
       CAST(SystemAgeAsOfReg AS DATE) AS SystemAgeAsOfReg,
       COALESCE(UPPER(TRIM(CountryID)), 'UNKNOWN') AS CountryID,
       COALESCE(UPPER(TRIM(Gender)), 'U') AS Gender
FROM Demographics
WHERE UserID IS NOT NULL;

CREATE OR REPLACE VIEW Clean_CashGames AS
SELECT CAST(UserID AS BIGINT) AS UserID,
       CAST(Date AS TIMESTAMP) AS ActivityDate,
       COALESCE(CAST(StakesC AS DOUBLE), 0.0) AS StakesC,
       COALESCE(CAST(WinningsC AS DOUBLE), 0.0) AS WinningsC
FROM CashGames
WHERE UserID IS NOT NULL AND Date IS NOT NULL;

CREATE OR REPLACE VIEW Clean_Tournaments AS
SELECT CAST(UserID AS BIGINT) AS UserID,
       CAST(Date AS TIMESTAMP) AS ActivityDate,
       COALESCE(CAST(Trnmnts AS DOUBLE), 0.0) AS Trnmnts,
       COALESCE(CAST(StakesT AS DOUBLE), 0.0) AS StakesT,
       COALESCE(CAST(WinningsT AS DOUBLE), 0.0) AS WinningsT
FROM Tournaments
WHERE UserID IS NOT NULL AND Date IS NOT NULL;

CREATE OR REPLACE VIEW Clean_Deposits AS
SELECT CAST(UserID AS BIGINT) AS UserID,
       CAST(ProcessDate AS TIMESTAMP) AS DepositDate,
       COALESCE(CAST(Amount AS DOUBLE), 0.0) AS DepositAmount
FROM Deposits
WHERE UserID IS NOT NULL AND ProcessDate IS NOT NULL;

CREATE OR REPLACE VIEW Clean_Withdrawals AS
SELECT CAST(UserID AS BIGINT) AS UserID,
       CAST(ProcessDate AS TIMESTAMP) AS WithdrawalDate,
       COALESCE(CAST(Amount AS DOUBLE), 0.0) AS WithdrawalAmount
FROM Withdrawals
WHERE UserID IS NOT NULL AND ProcessDate IS NOT NULL;
"""


def register_raw_views(connection: Any, data_directory: str | Path) -> dict[str, Path]:
    """Register each supplied CSV as a DuckDB view and return its resolved path."""
    root = Path(data_directory)
    paths = {name: root / filename for name, filename in RAW_TABLE_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required raw CSV files are missing. Place them in data/raw/: "
            + ", ".join(missing)
        )

    for table_name, path in paths.items():
        escaped_path = str(path.resolve()).replace("'", "''")
        connection.execute(
            f"CREATE OR REPLACE VIEW {table_name} "
            f"AS SELECT * FROM read_csv_auto('{escaped_path}')"
        )
    return paths


def initialize_clean_views(connection: Any) -> None:
    """Create standardized views for all five registered source tables."""
    connection.execute(CLEANING_SQL)
