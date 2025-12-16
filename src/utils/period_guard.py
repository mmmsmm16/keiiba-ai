"""
Period Guard Utility
年度スライスガード - Holdout(2025)データへのアクセスを制御

Usage:
    python script.py --start_year 2021 --end_year 2024
    python script.py --period screening  # 2024
    python script.py --period verification  # 2021-2023
    python script.py --period holdout --allow_holdout  # 2025 (Phase 8 only)
"""

import argparse
from dataclasses import dataclass
from typing import Optional, Tuple


# Period presets
PERIOD_PRESETS = {
    "screening": (2024, 2024),
    "verification": (2021, 2023),
    "holdout": (2025, 2025),
}

HOLDOUT_YEAR = 2025


@dataclass
class PeriodConfig:
    """Period configuration for evaluation"""
    start_year: int
    end_year: int
    allow_holdout: bool = False
    
    def validate(self) -> None:
        """Validate period configuration and block holdout if not allowed"""
        if self.end_year >= HOLDOUT_YEAR and not self.allow_holdout:
            raise ValueError(
                f"🚫 Holdout data ({HOLDOUT_YEAR}) is BLOCKED until Phase 8. "
                f"Requested period: {self.start_year}-{self.end_year}. "
                f"Use --allow_holdout flag explicitly in Phase 8."
            )
        
        if self.start_year > self.end_year:
            raise ValueError(
                f"Invalid period: start_year ({self.start_year}) > end_year ({self.end_year})"
            )
        
        if self.start_year < 2010 or self.end_year > 2030:
            raise ValueError(
                f"Period out of reasonable range: {self.start_year}-{self.end_year}"
            )
    
    def __post_init__(self):
        self.validate()
    
    @property
    def years(self) -> list:
        """Return list of years in the period"""
        return list(range(self.start_year, self.end_year + 1))
    
    def contains_holdout(self) -> bool:
        """Check if period contains holdout year"""
        return self.end_year >= HOLDOUT_YEAR
    
    def to_filter_expr(self, year_column: str = "year") -> str:
        """Generate filter expression for pandas/SQL"""
        return f"({year_column} >= {self.start_year}) & ({year_column} <= {self.end_year})"


def add_period_args(parser: argparse.ArgumentParser) -> None:
    """Add period-related arguments to an argument parser"""
    group = parser.add_mutually_exclusive_group(required=True)
    
    group.add_argument(
        "--period",
        type=str,
        choices=list(PERIOD_PRESETS.keys()),
        help="Period preset: screening (2024), verification (2021-2023), holdout (2025)"
    )
    
    group.add_argument(
        "--start_year",
        type=int,
        help="Start year (use with --end_year)"
    )
    
    parser.add_argument(
        "--end_year",
        type=int,
        help="End year (use with --start_year)"
    )
    
    parser.add_argument(
        "--allow_holdout",
        action="store_true",
        help="Allow access to holdout data (2025). USE ONLY IN PHASE 8."
    )


def parse_period_args(args: argparse.Namespace) -> PeriodConfig:
    """Parse period arguments and return PeriodConfig"""
    if args.period:
        start_year, end_year = PERIOD_PRESETS[args.period]
    else:
        if args.start_year is None or args.end_year is None:
            raise ValueError("Both --start_year and --end_year must be specified")
        start_year = args.start_year
        end_year = args.end_year
    
    return PeriodConfig(
        start_year=start_year,
        end_year=end_year,
        allow_holdout=args.allow_holdout
    )


def validate_period(
    start_year: int,
    end_year: int,
    allow_holdout: bool = False
) -> PeriodConfig:
    """
    Validate period and return PeriodConfig.
    This is the main guard function to be called by other scripts.
    
    Args:
        start_year: Start year of the period
        end_year: End year of the period
        allow_holdout: Whether to allow holdout data (Phase 8 only)
    
    Returns:
        PeriodConfig if valid
    
    Raises:
        ValueError: If period is invalid or holdout is blocked
    """
    return PeriodConfig(
        start_year=start_year,
        end_year=end_year,
        allow_holdout=allow_holdout
    )


def filter_dataframe_by_period(
    df,
    period: PeriodConfig,
    year_column: str = "year"
):
    """
    Filter a pandas DataFrame by period.
    
    Args:
        df: pandas DataFrame
        period: PeriodConfig instance
        year_column: Name of the year column
    
    Returns:
        Filtered DataFrame
    """
    return df[(df[year_column] >= period.start_year) & 
              (df[year_column] <= period.end_year)]


# CLI for testing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Period Guard - Validate and configure evaluation period"
    )
    add_period_args(parser)
    
    args = parser.parse_args()
    
    try:
        config = parse_period_args(args)
        print(f"✅ Period validated: {config.start_year}-{config.end_year}")
        print(f"   Years: {config.years}")
        print(f"   Contains holdout: {config.contains_holdout()}")
        print(f"   Holdout allowed: {config.allow_holdout}")
    except ValueError as e:
        print(f"❌ {e}")
        exit(1)
