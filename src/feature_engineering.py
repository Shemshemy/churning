"""Player-relative churn labels and leakage-controlled feature engineering."""

from pathlib import Path
from typing import Any

import pandas as pd


def create_player_temporal_target(
    connection: Any,
    observation_days: int = 30,
    target_days: int = 45,
) -> None:
    """Create a player-level target view from complete relative time windows."""
    if observation_days <= 0 or target_days <= 0:
        raise ValueError("Observation and target windows must be positive day counts.")

    connection.execute(
        f"""
        CREATE OR REPLACE VIEW Player_Temporal_Target AS
        WITH AllActivity AS (
            SELECT UserID, CAST(ActivityDate AS DATE) AS ActivityDate
            FROM Clean_CashGames
            UNION ALL
            SELECT UserID, CAST(ActivityDate AS DATE) AS ActivityDate
            FROM Clean_Tournaments
            UNION ALL
            SELECT UserID, CAST(DepositDate AS DATE) AS ActivityDate
            FROM Clean_Deposits
        ),
        DataCoverage AS (
            SELECT MAX(ActivityDate) AS latest_activity_date
            FROM AllActivity
        ),
        UserLifecycles AS (
            SELECT UserID,
                   MIN(ActivityDate) AS T_start,
                   MIN(ActivityDate) + INTERVAL '{observation_days} days' AS T_obs_end,
                   MIN(ActivityDate) + INTERVAL '{observation_days + target_days} days'
                       AS T_target_end
            FROM AllActivity
            GROUP BY UserID
        ),
        EligibleLifecycles AS (
            SELECT u.*
            FROM UserLifecycles u
            CROSS JOIN DataCoverage d
            WHERE u.T_target_end <= d.latest_activity_date
        ),
        PreCutoffBets AS (
            SELECT c.UserID, MAX(CAST(c.ActivityDate AS DATE)) AS last_obs_bet
            FROM Clean_CashGames c
            JOIN EligibleLifecycles u ON c.UserID = u.UserID
            WHERE CAST(c.ActivityDate AS DATE) <= u.T_obs_end
            GROUP BY c.UserID
        ),
        PostCutoffActivity AS (
            SELECT c.UserID
            FROM Clean_CashGames c
            JOIN EligibleLifecycles u ON c.UserID = u.UserID
            WHERE CAST(c.ActivityDate AS DATE) > u.T_obs_end
              AND CAST(c.ActivityDate AS DATE) <= u.T_target_end
            UNION
            SELECT t.UserID
            FROM Clean_Tournaments t
            JOIN EligibleLifecycles u ON t.UserID = u.UserID
            WHERE CAST(t.ActivityDate AS DATE) > u.T_obs_end
              AND CAST(t.ActivityDate AS DATE) <= u.T_target_end
            UNION
            SELECT d.UserID
            FROM Clean_Deposits d
            JOIN EligibleLifecycles u ON d.UserID = u.UserID
            WHERE CAST(d.DepositDate AS DATE) > u.T_obs_end
              AND CAST(d.DepositDate AS DATE) <= u.T_target_end
        )
        SELECT u.UserID, u.T_start, u.T_obs_end, u.T_target_end,
               COALESCE(
                   DATEDIFF('day', pb.last_obs_bet, u.T_obs_end),
                   999
               ) AS days_since_last_bet,
               CASE WHEN post.UserID IS NULL THEN 1 ELSE 0 END AS churned
        FROM EligibleLifecycles u
        LEFT JOIN PreCutoffBets pb ON u.UserID = pb.UserID
        LEFT JOIN PostCutoffActivity post ON u.UserID = post.UserID
        """
    )


def create_player_feature_matrix(connection: Any) -> None:
    """Create a feature view using only events from each player's observation window."""
    connection.execute(
        """
        CREATE OR REPLACE VIEW Player_Feature_Matrix AS
        WITH UserLifecycles AS (
            SELECT UserID, T_obs_end FROM Player_Temporal_Target
        ),
        Cash_Stats AS (
            SELECT c.UserID,
                   COUNT(DISTINCT CAST(c.ActivityDate AS DATE)) AS total_cash_sessions,
                   SUM(c.StakesC) AS total_stakes,
                   SUM(c.WinningsC) AS total_winnings
            FROM Clean_CashGames c
            JOIN UserLifecycles u ON c.UserID = u.UserID
            WHERE CAST(c.ActivityDate AS DATE) <= u.T_obs_end
            GROUP BY c.UserID
        ),
        Deposit_Stats AS (
            SELECT d.UserID,
                   COUNT(*) AS total_deposit_count,
                   SUM(d.DepositAmount) AS total_deposited,
                   MAX(CAST(d.DepositDate AS DATE)) AS LastDepositDate
            FROM Clean_Deposits d
            JOIN UserLifecycles u ON d.UserID = u.UserID
            WHERE CAST(d.DepositDate AS DATE) <= u.T_obs_end
            GROUP BY d.UserID
        )
        SELECT t.UserID,
               t.days_since_last_bet,
               t.churned,
               COALESCE(cs.total_cash_sessions, 0) AS total_cash_sessions,
               COALESCE(cs.total_stakes, 0.0) AS total_stakes,
               COALESCE(cs.total_winnings, 0.0) AS total_winnings,
               COALESCE(cs.total_winnings - cs.total_stakes, 0.0) AS net_cash_flow,
               COALESCE(ds.total_deposit_count, 0) AS total_deposit_count,
               COALESCE(ds.total_deposited, 0.0) AS total_deposited,
               COALESCE(
                   DATEDIFF('day', ds.LastDepositDate, t.T_obs_end),
                   999
               ) AS days_since_last_deposit,
               COALESCE(ds.total_deposited / (cs.total_stakes + 1.0), 0.0)
                   AS deposit_to_stake_ratio,
               COALESCE(cs.total_winnings / (cs.total_stakes + 1.0), 0.0)
                   AS win_rate_ratio
        FROM Player_Temporal_Target t
        LEFT JOIN Cash_Stats cs ON t.UserID = cs.UserID
        LEFT JOIN Deposit_Stats ds ON t.UserID = ds.UserID
        """
    )


def export_prepared_dataset(
    connection: Any,
    output_path: str | Path,
) -> pd.DataFrame:
    """Fetch the feature view, export it as Parquet, and return the DataFrame."""
    frame = connection.execute(
        "SELECT * FROM Player_Feature_Matrix"
    ).fetchdf()
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(destination, index=False)
    return frame
