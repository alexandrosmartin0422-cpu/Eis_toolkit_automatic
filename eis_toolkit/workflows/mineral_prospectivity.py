"""One-click mineral prospectivity mapping workflow.

This module wires together preprocessing, model training, prediction and
reporting into a single entry point. Given a DEM raster, a fault vector, a
geology vector and a deposit (e.g. gold occurrence) vector, it produces a
prospectivity map (PDF), a statistics workbook (XLSX) and a summary report
(DOCX).

The workflow can be triggered with a single command. Deposit points are used as
positive samples and randomly drawn background cells as negative samples. The
classifier is configurable (Random Forest, Gradient Boosting or Logistic
Regression) and any number of extra evidence rasters (e.g. slope, aspect,
geophysics) can be supplied in addition to the DEM, fault and geology layers.
"""

import os
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from beartype import beartype
from beartype.typing import Dict, List, Literal, Optional, Sequence, Tuple, Union
from rasterio import profiles

from eis_toolkit.exceptions import InvalidParameterValueException
from eis_toolkit.prediction.gradient_boosting import gradient_boosting_classifier_train
from eis_toolkit.prediction.logistic_regression import logistic_regression_train
from eis_toolkit.prediction.machine_learning_general import prepare_data_for_ml
from eis_toolkit.prediction.machine_learning_predict import predict_classifier
from eis_toolkit.prediction.random_forests import random_forest_classifier_train
from eis_toolkit.vector_processing.distance_computation import distance_computation
from eis_toolkit.vector_processing.rasterize_vector import rasterize_vector

Number = Union[int, float]

# GDAL reads shapefiles that are missing their .shx index when this is enabled.
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

ModelType = Literal["random_forest", "gradient_boosting", "logistic_regression"]


def _read_vector(path: Union[str, os.PathLike], fallback_crs: Optional[str] = None) -> gpd.GeoDataFrame:
    """Read a vector file, assigning a fallback CRS if the file has none."""
    gdf = gpd.read_file(path)
    if gdf.crs is None and fallback_crs is not None:
        gdf = gdf.set_crs(fallback_crs)
    return gdf


def _align_crs(gdf: gpd.GeoDataFrame, target_crs) -> gpd.GeoDataFrame:
    """Reproject a GeoDataFrame to the target CRS if needed."""
    if gdf.crs is None:
        return gdf.set_crs(target_crs)
    if gdf.crs != target_crs:
        return gdf.to_crs(target_crs)
    return gdf


def _profile_to_plain_dict(profile: Union[profiles.Profile, dict]) -> dict:
    """Return a writable raster profile dict for a single-band float32 raster."""
    out = dict(profile)
    out.update(count=1, dtype="float32", nodata=-9999.0)
    return out


def _write_raster(path: Union[str, os.PathLike], data: np.ndarray, profile: dict) -> None:
    """Write a single-band 2D array to a GeoTIFF using the given profile."""
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data.astype(profile["dtype"]), 1)


def _resample_raster_to_grid(src_path, ref_profile: dict, out_path) -> None:
    """Resample any raster onto the reference grid and write a single band."""
    from rasterio.warp import reproject
    from rasterio.enums import Resampling

    with rasterio.open(src_path) as src:
        destination = np.full((ref_profile["height"], ref_profile["width"]), ref_profile["nodata"], dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            dst_nodata=ref_profile["nodata"],
            resampling=Resampling.bilinear,
        )
    _write_raster(out_path, destination, ref_profile)


@beartype
def _build_evidence_rasters(
    dem_path: Union[str, os.PathLike],
    fault_path: Union[str, os.PathLike],
    geology_path: Union[str, os.PathLike],
    workdir: Union[str, os.PathLike],
    extra_raster_paths: Optional[Sequence[Union[str, os.PathLike]]] = None,
    fallback_crs: Optional[str] = None,
) -> Tuple[List[str], dict]:
    """Create aligned evidence rasters from the input layers.

    The DEM defines the reference grid. The fault vector is turned into a
    distance-to-fault raster and the geology vector is rasterized. Any extra
    rasters (e.g. slope, aspect, geophysics) are resampled onto the DEM grid.
    All evidence rasters share the DEM grid so they can be stacked for ML.

    Args:
        extra_raster_paths: Optional additional evidence rasters to include.
        fallback_crs: CRS assigned to vector inputs that lack a defined CRS
            (e.g. shapefiles without a .prj file).

    Returns:
        List of evidence raster filepaths and the reference profile dict.
    """
    with rasterio.open(dem_path) as dem:
        ref_profile = _profile_to_plain_dict(dem.profile)
        dem_data = dem.read(1).astype("float32")
        dem_crs = dem.crs

    workdir = Path(workdir)
    evidence_paths: List[str] = []

    # DEM evidence raster (copied into the working profile so the grid matches).
    dem_evidence = workdir / "evidence_dem.tif"
    _write_raster(dem_evidence, dem_data, ref_profile)
    evidence_paths.append(str(dem_evidence))

    # Distance-to-fault evidence raster.
    faults = _align_crs(_read_vector(fault_path, fallback_crs), dem_crs)
    fault_distance, _ = distance_computation(geodataframe=faults, raster_profile=ref_profile)
    fault_evidence = workdir / "evidence_fault_distance.tif"
    _write_raster(fault_evidence, fault_distance, ref_profile)
    evidence_paths.append(str(fault_evidence))

    # Geology evidence raster (rasterized class values).
    geology = _align_crs(_read_vector(geology_path, fallback_crs), dem_crs)
    value_column = _pick_geology_value_column(geology)
    geology_array = rasterize_vector(
        geodataframe=geology,
        raster_profile=ref_profile,
        value_column=value_column,
        default_value=1.0,
        fill_value=0.0,
    )
    geology_evidence = workdir / "evidence_geology.tif"
    _write_raster(geology_evidence, geology_array.astype("float32"), ref_profile)
    evidence_paths.append(str(geology_evidence))

    # Extra evidence rasters resampled onto the reference grid.
    for index, extra_path in enumerate(extra_raster_paths or []):
        name = Path(extra_path).stem
        out_path = workdir / f"evidence_extra_{index}_{name}.tif"
        _resample_raster_to_grid(extra_path, ref_profile, out_path)
        evidence_paths.append(str(out_path))

    return evidence_paths, ref_profile


def _pick_geology_value_column(geology: gpd.GeoDataFrame) -> Optional[str]:
    """Pick a numeric column to encode geology classes, else rasterize presence."""
    for column in geology.columns:
        if column == geology.geometry.name:
            continue
        if pd.api.types.is_numeric_dtype(geology[column]):
            return column
    return None


def _sample_background(
    deposit_labels: np.ndarray, n_positive: int, random_state: Optional[int]
) -> np.ndarray:
    """Add background (negative) samples to a 0/1 label array for training.

    Deposit cells are kept as positives. An equal number of non-deposit cells
    are flagged for use as negatives via a returned boolean training mask.
    """
    rng = np.random.default_rng(random_state)
    negative_pool = np.flatnonzero(deposit_labels == 0)
    n_negative = min(len(negative_pool), max(n_positive, 1))
    chosen = rng.choice(negative_pool, size=n_negative, replace=False)
    train_mask = deposit_labels == 1
    train_mask[chosen] = True
    return train_mask


def _train_model(
    model: ModelType,
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    random_state: Optional[int],
):
    """Dispatch to the requested classifier trainer."""
    metrics = ["accuracy", "precision", "recall", "f1"]
    if model == "random_forest":
        return random_forest_classifier_train(
            X=X, y=y, validation_method="split", metrics=metrics,
            n_estimators=n_estimators, random_state=random_state,
        )
    if model == "gradient_boosting":
        return gradient_boosting_classifier_train(
            X=X, y=y, validation_method="split", metrics=metrics,
            n_estimators=n_estimators, random_state=random_state,
        )
    if model == "logistic_regression":
        return logistic_regression_train(
            X=X, y=y, validation_method="split", metrics=metrics,
            random_state=random_state,
        )
    raise InvalidParameterValueException(f"Unknown model type: {model}")


def _filter_deposits(deposits: gpd.GeoDataFrame, commodity_filter: Optional[str]) -> gpd.GeoDataFrame:
    """Keep only deposits whose commodity column contains the filter substring."""
    if not commodity_filter:
        return deposits
    candidate_columns = [c for c in deposits.columns if "COMMODITY" in c.upper()]
    if not candidate_columns:
        return deposits
    mask = np.zeros(len(deposits), dtype=bool)
    for column in candidate_columns:
        mask |= deposits[column].astype(str).str.contains(commodity_filter, case=False, na=False)
    return deposits[mask]


@beartype
def run_mineral_prospectivity_workflow(
    dem_file: Union[str, os.PathLike],
    fault_file: Union[str, os.PathLike],
    geology_file: Union[str, os.PathLike],
    deposit_file: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    extra_rasters: Optional[Sequence[Union[str, os.PathLike]]] = None,
    model: ModelType = "random_forest",
    compare: bool = False,
    commodity_filter: Optional[str] = None,
    fallback_crs: Optional[str] = None,
    n_estimators: int = 100,
    random_state: Optional[int] = 42,
) -> Dict[str, str]:
    """Run the full mineral prospectivity mapping workflow.

    Args:
        dem_file: Filepath to the DEM raster. Defines the reference grid.
        fault_file: Filepath to the fault vector data.
        geology_file: Filepath to the geology vector data.
        deposit_file: Filepath to the deposit/occurrence point vector data
            (e.g. gold occurrences). Used as positive training samples.
        output_dir: Directory where the outputs are written. Created if missing.
        extra_rasters: Optional additional evidence rasters (e.g. slope, aspect,
            geophysics). Resampled onto the DEM grid.
        model: Classifier to use: "random_forest", "gradient_boosting" or
            "logistic_regression". Defaults to "random_forest". When ``compare``
            is True this is the primary model used for the main map and report.
        compare: If True, train all three classifiers on the same data, write a
            prospectivity raster per model and add a model comparison table to
            the outputs. Defaults to False.
        commodity_filter: If given, keep only deposits whose commodity attribute
            contains this text (e.g. "Au" for gold). Defaults to None (all).
        fallback_crs: CRS string (e.g. "EPSG:4283") assigned to vector inputs
            that lack a defined CRS, such as shapefiles missing a .prj file.
        n_estimators: Number of trees for tree-based models. Defaults to 100.
        random_state: Seed for reproducibility. Defaults to 42.

    Returns:
        Dictionary mapping output names ("map", "statistics", "report",
        "prospectivity_raster") to their written filepaths.

    Raises:
        InvalidParameterValueException: If the deposit data yields no positive
            samples on the reference grid.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as workdir:
        evidence_files, ref_profile = _build_evidence_rasters(
            dem_file, fault_file, geology_file, workdir,
            extra_raster_paths=extra_rasters, fallback_crs=fallback_crs,
        )

        # Rasterize deposits on the reference grid for use as labels.
        with rasterio.open(evidence_files[0]) as ref:
            ref_crs = ref.crs
        deposits = _align_crs(_read_vector(deposit_file, fallback_crs), ref_crs)
        deposits = _filter_deposits(deposits, commodity_filter)
        if deposits.empty:
            raise InvalidParameterValueException(
                "No deposits remain after filtering; check the commodity filter."
            )
        deposit_label_path = Path(workdir) / "deposits_label.tif"
        deposit_array = rasterize_vector(
            geodataframe=deposits, raster_profile=ref_profile, default_value=1.0, fill_value=0.0
        )
        _write_raster(deposit_label_path, deposit_array.astype("float32"), ref_profile)

        # Build X (all evidence) and y (deposit labels) on valid cells.
        X, y, reference_profile, nodata_mask = prepare_data_for_ml(
            feature_raster_files=evidence_files, label_file=str(deposit_label_path)
        )
        y = (y > 0).astype(int)

        n_positive = int(y.sum())
        if n_positive == 0:
            raise InvalidParameterValueException(
                "No deposit points fall on the DEM grid; check CRS and extent of the deposit data."
            )

        # Train on deposits + sampled background, then predict the whole grid.
        train_mask = _sample_background(y, n_positive, random_state)
        out_profile = _profile_to_plain_dict(reference_profile)
        result_paths: Dict[str, str] = {}

        models_to_run = list(ModelType.__args__) if compare else [model]
        comparison_rows = []
        primary = {}

        for current in models_to_run:
            trained_model, train_metrics = _train_model(
                current, X[train_mask], y[train_mask], n_estimators, random_state
            )
            _, probabilities = predict_classifier(
                data=X, model=trained_model, include_probabilities=True
            )
            prospectivity = _restore_to_grid(probabilities, nodata_mask, reference_profile)

            raster_name = "prospectivity.tif" if current == model else f"prospectivity_{current}.tif"
            raster_path = output_dir / raster_name
            _write_raster(raster_path, prospectivity, out_profile)
            result_paths[f"prospectivity_raster_{current}" if compare else "prospectivity_raster"] = str(raster_path)

            comparison_rows.append({"model": current, **{k: float(v) for k, v in train_metrics.items()}})

            if current == model:
                primary = dict(
                    prospectivity=prospectivity, train_metrics=train_metrics,
                    trained_model=trained_model, prospectivity_path=str(raster_path),
                )

        prospectivity = primary["prospectivity"]
        stats = _collect_statistics(
            prospectivity, nodata_mask, primary["train_metrics"], n_positive,
            int(train_mask.sum()), evidence_files, primary["trained_model"],
        )
        stats["summary"]["Model"] = model
        if compare:
            stats["comparison"] = comparison_rows

        from eis_toolkit.workflows.reporting import write_map_pdf, write_report_docx, write_statistics_xlsx

        map_path = output_dir / "Map.pdf"
        write_map_pdf(prospectivity, out_profile, deposits, str(map_path))

        xlsx_path = output_dir / "Statistics.xlsx"
        write_statistics_xlsx(stats, str(xlsx_path))

        report_inputs = {
            "DEM": str(dem_file),
            "Fault": str(fault_file),
            "Geology": str(geology_file),
            "Deposits": str(deposit_file),
            "Model": model,
        }
        for extra in extra_rasters or []:
            report_inputs[f"Extra: {Path(extra).name}"] = str(extra)

        docx_path = output_dir / "Report.docx"
        write_report_docx(stats, str(map_path), str(docx_path), inputs=report_inputs)

    outputs = {
        "map": str(map_path),
        "statistics": str(xlsx_path),
        "report": str(docx_path),
        "prospectivity_raster": primary["prospectivity_path"],
    }
    outputs.update(result_paths)
    return outputs


def _restore_to_grid(
    values: np.ndarray, nodata_mask: np.ndarray, reference_profile: Union[profiles.Profile, dict]
) -> np.ndarray:
    """Place predicted values back onto the full reference grid."""
    full = np.full(nodata_mask.shape, -9999.0, dtype="float32")
    full[~nodata_mask] = values.astype("float32")
    height = reference_profile["height"]
    width = reference_profile["width"]
    return full.reshape(height, width)


def _collect_statistics(
    prospectivity: np.ndarray,
    nodata_mask: np.ndarray,
    train_metrics: dict,
    n_positive: int,
    n_train: int,
    evidence_files: Sequence[str],
    model,
) -> dict:
    """Assemble the statistics dictionary used by the report writers."""
    valid = prospectivity[prospectivity != -9999.0]
    summary = {
        "Total cells": int(prospectivity.size),
        "Valid cells": int(valid.size),
        "Deposit (positive) samples": n_positive,
        "Training samples": n_train,
        "Min prospectivity": float(np.min(valid)) if valid.size else float("nan"),
        "Mean prospectivity": float(np.mean(valid)) if valid.size else float("nan"),
        "Max prospectivity": float(np.max(valid)) if valid.size else float("nan"),
        "High-potential cells (>0.5)": int(np.sum(valid > 0.5)),
    }

    feature_names = [Path(f).stem.replace("evidence_", "") for f in evidence_files]
    importances = getattr(model, "feature_importances_", None)
    importance = {}
    if importances is not None:
        for name, value in zip(feature_names, importances):
            importance[name] = float(value)

    return {
        "summary": summary,
        "metrics": {k: float(v) for k, v in train_metrics.items()},
        "feature_importance": importance,
    }
