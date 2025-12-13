import yaml
import logging
from pathlib import Path

from data.data_loader import DataLoader
from src.strategies.sma_strategy import SMAStrategy
from src.backtest.backtester import Backtester

# ---------------------------------------------------
# Setup logging
# ---------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/backtest.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Load configuration
# ---------------------------------------------------
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------
# Main pipeline
# ---------------------------------------------------
def main():

    logger.info("Starting algorithmic trading system demo run")

    # -------------------------------
    # Load config
    # -------------------------------
    config = load_config("config/settings.yaml")

    initial_capital = config["capital"]["initial_capital"]
    leverage = config["capital"]["leverage"]

    # -------------------------------
    # Load market data
    # -------------------------------
    data_loader = DataLoader(
        data_dir="data/raw",
        frequency=config["backtest"]["frequency"]
    )

    symbol = "SPY"

    market_data = data_loader.load_price_data(
        symbol=symbol,
        start_date="2018-01-01",
        end_date="2024-01-01"
    )

    logger.info(f"Loaded {len(market_data)} rows of data for {symbol}")

    # -------------------------------
    # Instantiate strategy
    # -------------------------------
    sma_params = config["strategy"]["sma"]

    strategy = SMAStrategy(
        fast_window=sma_params["fast_window"],
        slow_window=sma_params["slow_window"],
        position_size=sma_params["position_size"]
    )

    # -------------------------------
    # Instantiate backtester
    # -------------------------------
    backtester = Backtester(
        data=market_data,
        strategy=strategy,
        initial_capital=initial_capital,
        leverage=leverage,
        commission=config["costs"]["commission_per_trade"],
        slippage_bps=config["costs"]["slippage_bps"],
        transaction_cost_bps=config["costs"]["transaction_cost_bps"],
        allow_partial_fills=config["backtest"]["allow_partial_fills"]
    )

    # -------------------------------
    # Run backtest
    # -------------------------------
    results = backtester.run()

    logger.info("Backtest completed")

    # -------------------------------
    # Display results
    # -------------------------------
    print("\n===== BACKTEST SUMMARY =====")
    print(f"Final equity: £{results['final_equity']:.2f}")
    print(f"Total return: {results['total_return']:.2%}")
    print(f"Number of trades: {results['num_trades']}")
    print(f"Max drawdown: {results['max_drawdown']:.2%}")

    # -------------------------------
    # Save results
    # -------------------------------
    output_path = Path("reports/performance")
    output_path.mkdir(parents=True, exist_ok=True)

    results["equity_curve"].to_csv(output_path / "equity_curve.csv")
    results["trades"].to_csv(output_path / "trades.csv")

    logger.info("Results saved to reports/performance/")


if __name__ == "__main__":
    main()
