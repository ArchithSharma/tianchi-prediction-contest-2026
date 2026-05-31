#!/usr/bin/env python3
"""Assign shallow mainshocks to PB2002 tectonic plates and plot the result."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Iterable

import geopandas as gpd
import numpy as np
import pandas as pd



ROOT = Path("/Users/xusi/phdstudy/code/Prediction_contest")
MAINSHOCK_TABLE = ROOT / "outputs/shallow_regional_analysis/shallow_mainshock_aftershock_windows_depth_le_70_r200.parquet"
PLATES_GEOJSON = ROOT / "tectonicplates-master/GeoJSON/PB2002_plates.json"
BOUNDARIES_GEOJSON = ROOT / "tectonicplates-master/GeoJSON/PB2002_boundaries.json"
OUTPUT_DIR = ROOT / "outputs/plate_mainshock_analysis"
WINDOWS = ("T1", "T2", "T3")


def setup_plot_env(output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".mplconfig"))
    os.environ.setdefault("GMT_USERDIR", str(output_dir / ".gmt"))
    os.environ.setdefault("GMT_TMPDIR", str(output_dir / ".gmt_tmp"))
    for key in ["MPLCONFIGDIR", "GMT_USERDIR", "GMT_TMPDIR"]:
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def load_mainshocks(path: Path = MAINSHOCK_TABLE) -> pd.DataFrame:
    table = pd.read_parquet(path)
    table["mainshock_time_utc"] = pd.to_datetime(table["mainshock_time_utc"], utc=True, errors="coerce")
    for col in ["mainshock_latitude", "mainshock_longitude", "mainshock_depth_km", "mainshock_mag"]:
        table[col] = pd.to_numeric(table[col], errors="coerce")
    table = table.dropna(
        subset=["mainshock_time_utc", "mainshock_latitude", "mainshock_longitude", "mainshock_mag"]
    ).copy()
    return table


def load_plate_polygons(path: Path = PLATES_GEOJSON) -> gpd.GeoDataFrame:
    plates = gpd.read_file(path)
    plates = plates[["Code", "PlateName", "geometry"]].copy()
    plates = plates.set_crs("EPSG:4326", allow_override=True)
    return plates


def assign_plates(mainshocks: pd.DataFrame, plates: gpd.GeoDataFrame) -> pd.DataFrame:
    points = gpd.GeoDataFrame(
        mainshocks.copy(),
        geometry=gpd.points_from_xy(mainshocks["mainshock_longitude"], mainshocks["mainshock_latitude"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(points, plates, how="left", predicate="within")

    missing = joined["PlateName"].isna()
    if missing.any():
        fallback = gpd.sjoin(points.loc[missing], plates, how="left", predicate="intersects")
        joined.loc[missing, "Code"] = fallback["Code"].to_numpy()
        joined.loc[missing, "PlateName"] = fallback["PlateName"].to_numpy()

    joined["plate_code"] = joined["Code"].fillna("UN")
    joined["plate_name"] = joined["PlateName"].fillna("Unassigned")
    joined = pd.DataFrame(joined.drop(columns=["geometry", "index_right", "Code", "PlateName"], errors="ignore"))
    return joined


def add_plate_groups(table: pd.DataFrame, min_count: int = 30) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = table["plate_name"].value_counts().rename_axis("plate_name").reset_index(name="mainshock_count")
    keep = set(counts.loc[counts["mainshock_count"] >= min_count, "plate_name"])
    grouped = table.copy()
    grouped["plate_group"] = np.where(grouped["plate_name"].isin(keep), grouped["plate_name"], "Other")
    grouped_counts = (
        grouped["plate_group"]
        .value_counts()
        .rename_axis("plate_group")
        .reset_index(name="mainshock_count")
        .sort_values("mainshock_count", ascending=False)
        .reset_index(drop=True)
    )
    grouped_counts["is_other"] = grouped_counts["plate_group"].eq("Other")
    return grouped, grouped_counts


def analysis_tag(min_count: int) -> str:
    return f"depth_le_70_min{min_count}"


def plate_twindow_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    plate_counts = table["plate_group"].value_counts().to_dict()
    plate_order = table["plate_group"].value_counts().index.tolist()
    for plate_group in plate_order:
        group_table = table[table["plate_group"].eq(plate_group)]
        for window in WINDOWS:
            delay = pd.to_numeric(group_table[f"{window}_delta_hours"], errors="coerce").dropna()
            mag_delta = pd.to_numeric(group_table[f"{window}_mag_delta"], errors="coerce").dropna()
            rows.append(
                {
                    "plate_group": plate_group,
                    "window": window,
                    "mainshock_count": int(plate_counts.get(plate_group, 0)),
                    "target_count": int(delay.count()),
                    "target_coverage_pct": float(delay.count() / len(group_table) * 100.0),
                    "delay_mean_h": np.nan if delay.empty else float(delay.mean()),
                    "delay_median_h": np.nan if delay.empty else float(delay.median()),
                    "delay_q25_h": np.nan if delay.empty else float(delay.quantile(0.25)),
                    "delay_q75_h": np.nan if delay.empty else float(delay.quantile(0.75)),
                    "mag_delta_mean": np.nan if mag_delta.empty else float(mag_delta.mean()),
                    "mag_delta_median": np.nan if mag_delta.empty else float(mag_delta.median()),
                    "mag_delta_q25": np.nan if mag_delta.empty else float(mag_delta.quantile(0.25)),
                    "mag_delta_q75": np.nan if mag_delta.empty else float(mag_delta.quantile(0.75)),
                    "mag_delta_min": np.nan if mag_delta.empty else float(mag_delta.min()),
                    "mag_delta_max": np.nan if mag_delta.empty else float(mag_delta.max()),
                }
            )
    return pd.DataFrame(rows)


def color_map(categories: Iterable[str]) -> dict[str, str]:
    setup_plot_env(OUTPUT_DIR)
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    categories = list(categories)
    base = plt.colormaps["turbo"].resampled(max(len(categories), 2))
    colors = {
        category: mcolors.to_hex(base(i / max(len(categories) - 1, 1)))
        for i, category in enumerate(categories)
    }
    colors["Other"] = "#8A8F98"
    return colors


def plot_boundaries(fig, boundaries: gpd.GeoDataFrame, pen: str = "0.35p,#4B5563") -> None:
    for geom in boundaries.geometry:
        if geom is None or geom.is_empty:
            continue
        geoms = geom.geoms if geom.geom_type.startswith("Multi") else [geom]
        for item in geoms:
            if item.geom_type not in {"LineString", "LinearRing"}:
                continue
            x, y = item.xy
            fig.plot(x=list(x), y=list(y), pen=pen)


def plot_plate_map(
    table: pd.DataFrame,
    counts: pd.DataFrame,
    tag: str,
    output_dir: Path = OUTPUT_DIR,
    boundaries_path: Path = BOUNDARIES_GEOJSON,
) -> Path:
    setup_plot_env(output_dir)
    import pygmt

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"pygmt_mainshock_plate_groups_{tag}.png"
    boundaries = gpd.read_file(boundaries_path)
    colors = color_map(counts["plate_group"])

    with pygmt.config(FONT="8p,Helvetica", MAP_FRAME_TYPE="plain"):
        fig = pygmt.Figure()
        fig.basemap(
            region="g",
            projection="W180/7.7i",
            frame=["xaf60", "yaf30", "+tShallow M>6 Mainshocks by PB2002 Plate"],
        )
        fig.coast(
            land="#F3EBDD",
            water="#D8ECF4",
            shorelines="0.35p,#69737A",
            borders=["1/0.25p,#A1A1AA"],
        )
        plot_boundaries(fig, boundaries, pen="0.45p,#52525B")

        label_groups = set(counts.head(12)["plate_group"])
        if "Other" in set(counts["plate_group"]):
            label_groups.add("Other")
        for _, row in counts.iterrows():
            group = row["plate_group"]
            subset = table[table["plate_group"].eq(group)]
            if subset.empty:
                continue
            label = f"{group} ({int(row['mainshock_count'])})" if group in label_groups else None
            fig.plot(
                x=subset["mainshock_longitude"],
                y=subset["mainshock_latitude"],
                style="c0.07c",
                fill=colors[group],
                pen="0.12p,#111827",
                transparency=18,
                label=label,
            )
        fig.legend(position="JBR+jBR+o0.15i/0.15i", box="+gwhite@15+p0.4p,#6B7280")
        fig.savefig(path, dpi=240)
    return path


def plot_plate_counts(counts: pd.DataFrame, tag: str, output_dir: Path = OUTPUT_DIR) -> Path:
    setup_plot_env(output_dir)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    top = counts.head(24).copy()
    fig, ax = plt.subplots(figsize=(12, 7.2))
    colors = ["#8A8F98" if name == "Other" else "#2C7FB8" for name in top["plate_group"]]
    ax.barh(top["plate_group"], top["mainshock_count"], color=colors, edgecolor="#111827", linewidth=0.5)
    ax.invert_yaxis()
    ax.set_xlabel("M>6 shallow mainshocks")
    ax.set_title("Mainshock counts by PB2002 plate group")
    ax.grid(axis="x", color="#E5E7EB")
    for y, value in enumerate(top["mainshock_count"]):
        ax.text(value + 2, y, str(int(value)), va="center", fontsize=8)
    fig.tight_layout()
    path = output_dir / f"mainshock_plate_group_counts_{tag}.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plate_order_with_data(table: pd.DataFrame) -> list[str]:
    return table["plate_group"].value_counts().index.tolist()


def plot_twindow_boxplots(
    table: pd.DataFrame,
    value_suffix: str,
    ylabel: str,
    title: str,
    filename: str,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    setup_plot_env(output_dir)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plate_order = plate_order_with_data(table)
    fig, axes = plt.subplots(1, 3, figsize=(17, 6.4), sharey=False)
    for ax, window in zip(axes, WINDOWS):
        data = [
            pd.to_numeric(table.loc[table["plate_group"].eq(plate), f"{window}_{value_suffix}"], errors="coerce")
            .dropna()
            .to_numpy()
            for plate in plate_order
        ]
        non_empty_plates = [plate for plate, values in zip(plate_order, data) if len(values) > 0]
        non_empty_data = [values for values in data if len(values) > 0]
        box = ax.boxplot(non_empty_data, patch_artist=True, showfliers=False)
        for patch in box["boxes"]:
            patch.set_facecolor("#2C7FB8" if value_suffix == "delta_hours" else "#D84A3A")
            patch.set_alpha(0.65)
        ax.set_title(window)
        ax.set_xticklabels(non_empty_plates, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="#E5E7EB")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_twindow_heatmap(
    summary: pd.DataFrame,
    value_col: str,
    title: str,
    filename: str,
    output_dir: Path = OUTPUT_DIR,
) -> Path:
    setup_plot_env(output_dir)
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    plate_order = (
        summary[["plate_group", "mainshock_count"]]
        .drop_duplicates()
        .sort_values("mainshock_count", ascending=False)["plate_group"]
        .tolist()
    )
    matrix = summary.pivot(index="plate_group", columns="window", values=value_col).reindex(plate_order)
    fig, ax = plt.subplots(figsize=(6.8, max(4.8, 0.34 * len(matrix))))
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", aspect="auto")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, shrink=0.82)
    fig.tight_layout()
    path = output_dir / filename
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def run_analysis(
    output_dir: Path = OUTPUT_DIR,
    min_count: int = 30,
    mainshock_path: Path = MAINSHOCK_TABLE,
    plates_path: Path = PLATES_GEOJSON,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = analysis_tag(min_count)
    mainshocks = load_mainshocks(mainshock_path)
    plates = load_plate_polygons(plates_path)
    assigned = assign_plates(mainshocks, plates)
    grouped, counts = add_plate_groups(assigned, min_count=min_count)
    twindow_summary = plate_twindow_summary(grouped)

    assigned_path = output_dir / f"mainshock_plate_assignments_{tag}.parquet"
    assigned_csv_path = output_dir / f"mainshock_plate_assignments_{tag}.csv"
    counts_path = output_dir / f"mainshock_plate_group_counts_{tag}.csv"
    raw_counts_path = output_dir / "mainshock_plate_raw_counts_depth_le_70.csv"
    twindow_summary_path = output_dir / f"plate_twindow_delta_summary_{tag}.csv"
    grouped.to_parquet(assigned_path, index=False)
    grouped.to_csv(assigned_csv_path, index=False)
    counts.to_csv(counts_path, index=False)
    twindow_summary.to_csv(twindow_summary_path, index=False)
    grouped["plate_name"].value_counts().rename_axis("plate_name").reset_index(name="mainshock_count").to_csv(
        raw_counts_path,
        index=False,
    )

    map_path = plot_plate_map(grouped, counts, tag=tag, output_dir=output_dir)
    count_plot_path = plot_plate_counts(counts, tag=tag, output_dir=output_dir)
    delay_boxplot_path = plot_twindow_boxplots(
        grouped,
        value_suffix="delta_hours",
        ylabel="Delay hours",
        title="T1/T2/T3 delay distribution by PB2002 plate group",
        filename=f"plate_twindow_delay_boxplots_{tag}.png",
        output_dir=output_dir,
    )
    mag_delta_boxplot_path = plot_twindow_boxplots(
        grouped,
        value_suffix="mag_delta",
        ylabel="M_aftershock - M_mainshock",
        title="T1/T2/T3 magnitude difference distribution by PB2002 plate group",
        filename=f"plate_twindow_mag_delta_boxplots_{tag}.png",
        output_dir=output_dir,
    )
    delay_heatmap_path = plot_twindow_heatmap(
        twindow_summary,
        value_col="delay_median_h",
        title="Median T-window delay hours by plate group",
        filename=f"plate_twindow_delay_median_heatmap_{tag}.png",
        output_dir=output_dir,
    )
    mag_delta_heatmap_path = plot_twindow_heatmap(
        twindow_summary,
        value_col="mag_delta_median",
        title="Median T-window ΔM by plate group",
        filename=f"plate_twindow_mag_delta_median_heatmap_{tag}.png",
        output_dir=output_dir,
    )

    metadata = {
        "mainshock_table": str(mainshock_path),
        "plates_geojson": str(plates_path),
        "boundaries_geojson": str(BOUNDARIES_GEOJSON),
        "min_count_for_named_plate_group": min_count,
        "mainshock_rows": int(len(grouped)),
        "raw_plate_count": int(grouped["plate_name"].nunique()),
        "grouped_plate_count": int(grouped["plate_group"].nunique()),
        "other_rows": int(grouped["plate_group"].eq("Other").sum()),
        "assigned_parquet": str(assigned_path),
        "assigned_csv": str(assigned_csv_path),
        "counts_csv": str(counts_path),
        "raw_counts_csv": str(raw_counts_path),
        "twindow_summary_csv": str(twindow_summary_path),
        "map_png": str(map_path),
        "count_plot_png": str(count_plot_path),
        "delay_boxplot_png": str(delay_boxplot_path),
        "mag_delta_boxplot_png": str(mag_delta_boxplot_path),
        "delay_heatmap_png": str(delay_heatmap_path),
        "mag_delta_heatmap_png": str(mag_delta_heatmap_path),
    }
    metadata_path = output_dir / "plate_mainshock_analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "assigned": grouped,
        "counts": counts,
        "twindow_summary": twindow_summary,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "map_path": map_path,
        "count_plot_path": count_plot_path,
        "delay_boxplot_path": delay_boxplot_path,
        "mag_delta_boxplot_path": mag_delta_boxplot_path,
        "delay_heatmap_path": delay_heatmap_path,
        "mag_delta_heatmap_path": mag_delta_heatmap_path,
    }


if __name__ == "__main__":
    results = run_analysis()
    print(results["metadata_path"])
    print(results["map_path"])
