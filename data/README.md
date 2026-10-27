# Data dictionary and raw data setup

The notebook expects the original CSV exports in `data/raw/`. The source files
contain player-level and transaction-level information and are intentionally
excluded from version control by the root `.gitignore`.

## Setup

Obtain the five source files through the project's authorized data provider and
place them in `data/raw/` using these exact filenames:

1. `SecondSeshData1Demographics.csv`
2. `SecondSeshData2CashGames.csv`
3. `SecondSeshData3Tournaments.csv`
4. `SecondSeshData4Deposits.csv`
5. `SecondSeshData5Withdrawals.csv`

Do not upload or commit raw player-level data. The notebook and `src/data_cleaning.py`
raise a clear missing-file error if any expected CSV is absent.

## Source schemas

| File | Grain | Fields |
|---|---|---|
| Demographics | One record per player | `UserID` (player key), `SystemAgeAsOfReg` (registration/system age date), `Gender`, `CountryID` |
| Cash games | Player cash-game activity record | `UserID`, `Date`, `Windows`, `StakesC`, `WinningsC` |
| Tournaments | Player tournament activity record | `UserID`, `Date`, `Trnmnts`, `StakesT`, `WinningsT` |
| Deposits | Deposit transaction | `UserID`, `DepositID`, `SummaryDate`, `ProcessDate`, `ProcessTime`, `PayMeth`, `PayMethCat`, `CardType`, `Amount`, `Status` |
| Withdrawals | Withdrawal transaction | `UserID`, `WithdrawalID`, `SummaryDate`, `ProcessDate`, `ProcessTime`, `PayMeth`, `PayMethCat`, `CardType`, `Amount`, `Status` |

Identifiers and dates are normalized in DuckDB views. Missing numeric transaction
amounts are set to zero for aggregation; missing category values are represented
with explicit unknown labels.

## Derived dataset

The prepared Parquet file in `artifacts/` contains one row per eligible player.
The target uses a 30-day observation window and a subsequent 45-day outcome
window. Only histories with complete follow-up are retained. The model features
currently summarize cash-game and deposit activity; tournament activity is used
to determine the outcome, while withdrawals and demographics are profiled but
not currently used as predictors.
