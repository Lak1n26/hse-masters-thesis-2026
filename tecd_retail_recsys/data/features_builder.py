import numpy as np
import pandas as pd


class CandidateFeatureBuilder:
    """
    Строит фичи для пар (user_id, item_id) по всей доступной истории.

    Поддерживаемые группы:
    - "user"
    - "user_item"
    - "item"
    """

    VALID_GROUPS = {"user", "user_item", "item"}

    def __init__(self, events_df: pd.DataFrame):
        self.events = self._prepare_events(events_df)

        self.user_features_ = None
        self.user_item_features_ = None
        self.item_features_ = None

    @staticmethod
    def _prepare_events(events: pd.DataFrame) -> pd.DataFrame:
        df = events.copy()

        required_cols = [
            "timestamp", "user_id", "item_id", "day",
            "item_brand_id", "item_category", "item_subcategory", "item_price"
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in events_df: {missing}")

        df["timestamp"] = df["timestamp"].astype("int64")
        df["user_id"] = df["user_id"].astype("int64")
        df["item_id"] = df["item_id"].astype("int64")
        df["day"] = df["day"].astype("int64")
        df["item_price"] = pd.to_numeric(df["item_price"], errors="coerce").fillna(0).astype("float32")

        for col in ["item_brand_id", "item_category", "item_subcategory", "subdomain", "os"]:
            if col in df.columns:
                df[col] = df[col].astype("category")

        df = df.sort_values(["timestamp", "user_id", "item_id"]).reset_index(drop=True)
        return df

    @staticmethod
    def _validate_candidates(candidates_df: pd.DataFrame) -> pd.DataFrame:
        df = candidates_df.copy()

        required_cols = ["user_id", "item_id"]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in candidates_df: {missing}")

        df["user_id"] = df["user_id"].astype("int64")
        df["item_id"] = df["item_id"].astype("int64")
        return df

    def _build_user_features(self) -> pd.DataFrame:
        events = self.events

        user_agg = (
            events.groupby("user_id", observed=True)
            .agg(
                user_events_total=("item_id", "size"),
                user_unique_items=("item_id", "nunique"),
                user_unique_brands=("item_brand_id", "nunique"),
                user_unique_categories=("item_category", "nunique"),
                user_unique_subcategories=("item_subcategory", "nunique"),
                user_first_day=("day", "min"),
                user_last_day=("day", "max"),
                user_first_ts=("timestamp", "min"),
                user_last_ts=("timestamp", "max"),
                user_avg_price=("item_price", "mean"),
                user_median_price=("item_price", "median"),
                user_min_price=("item_price", "min"),
                user_max_price=("item_price", "max"),
                user_price_std=("item_price", "std"),
            )
            .reset_index()
        )

        user_agg["user_active_days"] = (
            user_agg["user_last_day"] - user_agg["user_first_day"] + 1
        ).astype("float32")

        user_agg["user_events_per_active_day"] = (
            user_agg["user_events_total"] / user_agg["user_active_days"].replace(0, np.nan)
        ).fillna(0).astype("float32")

        user_agg["user_repeat_item_ratio"] = (
            user_agg["user_events_total"] / user_agg["user_unique_items"].replace(0, np.nan)
        ).fillna(0).astype("float32")

        num_cols = [
            "user_events_total", "user_unique_items", "user_unique_brands",
            "user_unique_categories", "user_unique_subcategories",
            "user_avg_price", "user_median_price", "user_min_price",
            "user_max_price", "user_price_std", "user_active_days",
            "user_events_per_active_day", "user_repeat_item_ratio"
        ]
        user_agg[num_cols] = user_agg[num_cols].fillna(0)

        return user_agg

    def _build_user_item_features(self) -> pd.DataFrame:
        events = self.events

        ui = (
            events.groupby(["user_id", "item_id"], observed=True)
            .agg(
                user_item_interactions=("timestamp", "size"),
                user_item_first_day=("day", "min"),
                user_item_last_day=("day", "max"),
                user_item_first_ts=("timestamp", "min"),
                user_item_last_ts=("timestamp", "max"),
            )
            .reset_index()
        )

        ui["user_item_active_span_days"] = (
            ui["user_item_last_day"] - ui["user_item_first_day"] + 1
        ).astype("float32")

        ui["user_item_repeat_flag"] = (ui["user_item_interactions"] > 0).astype("int8")
        ui["top_personal_score"] = ui["user_item_interactions"].astype("float32")

        return ui

    def _build_item_features(self) -> pd.DataFrame:
        events = self.events

        item_agg = (
            events.groupby("item_id", observed=True)
            .agg(
                item_events_total=("user_id", "size"),
                item_unique_users=("user_id", "nunique"),
                item_first_day=("day", "min"),
                item_last_day=("day", "max"),
                item_first_ts=("timestamp", "min"),
                item_last_ts=("timestamp", "max"),
                item_avg_price=("item_price", "mean"),
                item_median_price=("item_price", "median"),
                item_min_price=("item_price", "min"),
                item_max_price=("item_price", "max"),
                item_price_std=("item_price", "std"),
            )
            .reset_index()
        )

        item_agg["item_active_days"] = (
            item_agg["item_last_day"] - item_agg["item_first_day"] + 1
        ).astype("float32")

        item_agg["item_events_per_active_day"] = (
            item_agg["item_events_total"] / item_agg["item_active_days"].replace(0, np.nan)
        ).fillna(0).astype("float32")

        item_agg["item_repeat_user_ratio"] = (
            item_agg["item_events_total"] / item_agg["item_unique_users"].replace(0, np.nan)
        ).fillna(0).astype("float32")

        item_last_attrs = (
            events.sort_values(["item_id", "timestamp"])
            .groupby("item_id", as_index=False)
            .last()[["item_id", "item_brand_id", "item_category", "item_subcategory", "item_price"]]
            .rename(columns={"item_price": "item_last_price"})
        )

        item_agg = item_agg.merge(item_last_attrs, on="item_id", how="left")

        num_cols = [
            "item_events_total", "item_unique_users",
            "item_avg_price", "item_median_price", "item_min_price",
            "item_max_price", "item_price_std", "item_active_days",
            "item_events_per_active_day", "item_repeat_user_ratio", "item_last_price"
        ]
        item_agg[num_cols] = item_agg[num_cols].fillna(0)

        return item_agg

    def fit(self, feature_groups=("user", "user_item", "item")):
        feature_groups = set(feature_groups)
        unknown = feature_groups - self.VALID_GROUPS
        if unknown:
            raise ValueError(f"Unknown feature groups: {sorted(unknown)}")

        if "user" in feature_groups:
            self.user_features_ = self._build_user_features()

        if "user_item" in feature_groups:
            self.user_item_features_ = self._build_user_item_features()

        if "item" in feature_groups:
            self.item_features_ = self._build_item_features()

        return self

    def transform(self, candidates_df: pd.DataFrame, feature_groups=("user", "user_item", "item")) -> pd.DataFrame:
        cands = self._validate_candidates(candidates_df)

        feature_groups = set(feature_groups)
        unknown = feature_groups - self.VALID_GROUPS
        if unknown:
            raise ValueError(f"Unknown feature groups: {sorted(unknown)}")

        out = cands.copy()

        if "user" in feature_groups:
            if self.user_features_ is None:
                self.user_features_ = self._build_user_features()
            out = out.merge(self.user_features_, on="user_id", how="left")

        if "user_item" in feature_groups:
            if self.user_item_features_ is None:
                self.user_item_features_ = self._build_user_item_features()
            out = out.merge(self.user_item_features_, on=["user_id", "item_id"], how="left")

        if "item" in feature_groups:
            if self.item_features_ is None:
                self.item_features_ = self._build_item_features()
            out = out.merge(self.item_features_, on="item_id", how="left")

        if {"item_last_price", "user_avg_price"}.issubset(out.columns):
            out["item_price_vs_user_avg"] = (
                out["item_last_price"] - out["user_avg_price"]
            ).astype("float32")

        if {"item_last_price", "user_avg_price", "user_price_std"}.issubset(out.columns):
            out["item_price_z_vs_user"] = (
                (out["item_last_price"] - out["user_avg_price"]) / (out["user_price_std"] + 1e-6)
            ).astype("float32")

        if {"user_item_interactions", "user_events_total"}.issubset(out.columns):
            out["user_item_share"] = (
                out["user_item_interactions"] / (out["user_events_total"] + 1e-6)
            ).astype("float32")

        if {"item_events_total", "item_unique_users"}.issubset(out.columns):
            out["item_popularity_per_user"] = (
                out["item_events_total"] / (out["item_unique_users"] + 1e-6)
            ).astype("float32")

        if "user_events_total" in out.columns:
            out["is_cold_user"] = (out["user_events_total"].fillna(0) <= 0).astype("int8")
        if "item_events_total" in out.columns:
            out["is_cold_item"] = (out["item_events_total"].fillna(0) <= 0).astype("int8")
        if "user_item_interactions" in out.columns:
            out["is_new_item_for_user"] = (out["user_item_interactions"].fillna(0) <= 0).astype("int8")

        num_cols = out.select_dtypes(include=[np.number]).columns.tolist()
        out[num_cols] = out[num_cols].fillna(0)

        return out

    def fit_transform(self, candidates_df: pd.DataFrame, feature_groups=("user", "user_item", "item")) -> pd.DataFrame:
        self.fit(feature_groups=feature_groups)
        return self.transform(candidates_df, feature_groups=feature_groups)