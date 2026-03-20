"""
gemini_data.py
==============
Real-time agricultural commodity price data fetcher using the Google Gemini API.

This module supplements the data.gov.in (agmarknet) dataset with live price
intelligence retrieved via Gemini's language model. Uses Pydantic models to
define structured output schemas so the API returns guaranteed well-formed JSON.
"""

import os
import sys
import logging
from datetime import datetime
from typing import List, Optional

import pandas as pd
from google import genai
from pydantic import BaseModel, Field

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured Gemini output
# ---------------------------------------------------------------------------
class PriceRecord(BaseModel):
    """A single commodity price record from Gemini."""
    date: str = Field(description="Date in YYYY-MM-DD format.")
    state: str = Field(description="State name.")
    district: str = Field(description="District name.")
    commodity: str = Field(description="Commodity name.")
    min_price: Optional[float] = Field(description="Minimum price in INR per quintal.")
    max_price: Optional[float] = Field(description="Maximum price in INR per quintal.")
    modal_price: Optional[float] = Field(description="Modal (most common) price in INR per quintal.")
    source: str = Field(default="gemini_realtime", description="Data source identifier.")


class PriceResponse(BaseModel):
    """Top-level response containing a list of price records."""
    records: List[PriceRecord] = Field(description="List of commodity price records.")


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
_PROMPT_TEMPLATE = """
You are an agricultural market data assistant. Your task is to provide the LATEST
available wholesale / mandi market prices for the commodities and state listed below.

State: {state}
Districts: {districts}
Commodities: {commodities}
Date: {date}

Provide one entry per (district × commodity) combination using the most recent
publicly known prices. If data for a specific district is unavailable use the
state-level average. Set the source field to "gemini_realtime" for all records.
"""


class GeminiDataClient:
    """
    Fetches real-time commodity price data via the Google Gemini API.

    Uses Pydantic-based structured output so the API returns guaranteed
    well-formed JSON matching the PriceResponse schema.

    Usage
    -----
    >>> config = load_config()
    >>> client = GeminiDataClient(config)
    >>> df = client.fetch_realtime_data()
    >>> print(df.head())
    """

    def __init__(self, config: dict):
        self.config = config
        self.api_key: str = config["api_keys"].get("gemini", "")
        self.state: str = config["region"]["state"]
        self.districts: list = config["region"]["districts"]
        self.commodities: list = config["commodities"]
        # Ordered list of model candidates — first usable one wins
        self.model_candidates: list = [
            "gemini-2.0-flash",
            "gemini-2.5-flash",
            "gemini-2.5-pro",
        ]
        self.model_name: str = self.model_candidates[0]  # updated after init
        self._client = None  # lazy-initialised

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_realtime_data(self, save: bool = True) -> pd.DataFrame:
        """
        Query Gemini for today's market prices and return a DataFrame.

        Parameters
        ----------
        save : bool
            If True the result is also written to data/raw/gemini_realtime_data.csv.

        Returns
        -------
        pd.DataFrame  (empty on failure)
        """
        if not self.api_key:
            logger.warning("No Gemini API key configured. Skipping real-time fetch.")
            return pd.DataFrame()

        try:
            self._init_client()
            today = datetime.now().strftime("%Y-%m-%d")
            prompt = _PROMPT_TEMPLATE.format(
                state=self.state,
                districts=", ".join(self.districts),
                commodities=", ".join(self.commodities),
                date=today,
            )

            logger.info("Querying Gemini API for real-time price data...")

            # Use structured output with Pydantic schema + JSON response config
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "response_json_schema": PriceResponse.model_json_schema(),
                },
            )

            # Parse the structured JSON response directly via Pydantic
            price_response = PriceResponse.model_validate_json(response.text)
            df = self._build_dataframe(price_response, today)

            if df.empty:
                logger.warning("Gemini returned an empty response.")
            else:
                logger.info(f"Gemini returned {len(df)} price records for {today}.")
                if save:
                    self._save(df)

            return df

        except Exception as exc:
            logger.error(f"Gemini API call failed: {exc}")
            return pd.DataFrame()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_client(self):
        """
        Initialise the Gemini client (new google.genai SDK).

        Tries each candidate model in order and picks the first one that works.
        This handles quota limits or deprecation of specific model versions.
        """
        if self._client is not None:
            return

        last_err = None

        for candidate in self.model_candidates:
            try:
                client = genai.Client(api_key=self.api_key)
                # Lightweight probe to confirm the model responds
                client.models.generate_content(
                    model=candidate,
                    contents="ping",
                )
                self._client = client
                self.model_name = candidate
                logger.info(f"Gemini model selected: {candidate}")
                return
            except Exception as e:
                logger.warning(f"Model {candidate!r} not usable: {e}")
                last_err = e

        # None of the candidates worked — re-raise the last error
        raise RuntimeError(
            f"No usable Gemini model found from candidates {self.model_candidates}. "
            f"Last error: {last_err}"
        )

    def _build_dataframe(self, price_response: PriceResponse, fallback_date: str) -> pd.DataFrame:
        """
        Convert the Pydantic PriceResponse into a DataFrame.

        Since the API returns structured JSON matching the schema, no manual
        regex parsing or JSON extraction is needed.
        """
        if not price_response.records:
            return pd.DataFrame()

        records = []
        for rec in price_response.records:
            records.append({
                "date": rec.date or fallback_date,
                "state": rec.state,
                "district": rec.district,
                "commodity": rec.commodity,
                "min_price": rec.min_price,
                "max_price": rec.max_price,
                "modal_price": rec.modal_price,
                "source": rec.source or "gemini_realtime",
            })

        df = pd.DataFrame(records)

        # Coerce numeric columns
        for price_col in ["min_price", "max_price", "modal_price"]:
            df[price_col] = pd.to_numeric(df[price_col], errors="coerce")

        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")

        return df

    def _save(self, df: pd.DataFrame, filename: str = "gemini_realtime_data.csv"):
        path = os.path.join(self.config["paths"]["raw_data"], filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"Gemini real-time data saved to {path}")


# ---------------------------------------------------------------------------
# Standalone execution for quick testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    cfg = load_config()
    client = GeminiDataClient(cfg)
    result_df = client.fetch_realtime_data(save=True)

    if not result_df.empty:
        print("\n=== Gemini Real-Time Price Data ===")
        print(result_df.to_string(index=False))
    else:
        print("No data returned from Gemini.")
