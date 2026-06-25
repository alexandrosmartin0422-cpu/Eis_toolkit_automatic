"""Build an interactive (Leaflet/Folium) web map from workflow outputs.

Takes a prospectivity raster plus deposit and fault vectors and produces a
single self-contained ``index.html`` (with a PNG overlay) that can be published
on GitHub Pages so anyone can pan/zoom the result in a browser. No server or
plugin is required by the viewer.

The map shows three layers, toggleable via a layer control:
- Gold prospectivity (semi-transparent raster overlay, magma colormap)
- Deposits (yellow circle markers)
- Faults (red; thrust/layered faults dashed)
"""

import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from beartype import beartype
from beartype.typing import Optional, Union

from eis_toolkit.workflows.reporting import _fault_lines, _split_faults

WEB_CRS = "EPSG:4326"


def _reproject_to_wgs84(raster_path, max_pixels: int = 2_000_000):
    """Reproject a raster to WGS84, downsampling so it is web-friendly.

    Returns (array, (south, west, north, east) bounds) with nodata as NaN.
    """
    from rasterio.warp import calculate_default_transform, reproject, Resampling

    with rasterio.open(raster_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, WEB_CRS, src.width, src.height, *src.bounds
        )
        # Downsample if the reprojected grid would be too large for the browser.
        scale = (width * height / max_pixels) ** 0.5
        if scale > 1:
            width = int(width / scale)
            height = int(height / scale)
            transform, width, height = calculate_default_transform(
                src.crs, WEB_CRS, src.width, src.height, *src.bounds,
                dst_width=width, dst_height=height,
            )
        destination = np.full((height, width), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=src.nodata,
            dst_transform=transform,
            dst_crs=WEB_CRS,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        west, north = transform * (0, 0)
        east, south = transform * (width, height)
    return destination, (south, west, north, east)


def _array_to_png(array: np.ndarray, png_path) -> None:
    """Write a 0..1 float array to an RGBA PNG using the magma colormap."""
    import matplotlib.cm as cm
    from PIL import Image

    valid = np.isfinite(array)
    normalized = np.clip(np.nan_to_num(array, nan=0.0), 0.0, 1.0)
    rgba = (cm.magma(normalized) * 255).astype("uint8")
    rgba[..., 3] = np.where(valid, 200, 0).astype("uint8")  # transparent nodata
    Image.fromarray(rgba, mode="RGBA").save(png_path)


@beartype
def build_web_map(
    prospectivity_raster: Union[str, os.PathLike],
    output_dir: Union[str, os.PathLike],
    deposit_file: Optional[Union[str, os.PathLike]] = None,
    fault_file: Optional[Union[str, os.PathLike]] = None,
    commodity_filter: Optional[str] = None,
    fallback_crs: Optional[str] = None,
    title: str = "Mineral Prospectivity Map",
) -> str:
    """Build an interactive web map and write index.html to output_dir.

    Args:
        prospectivity_raster: Filepath to the prospectivity GeoTIFF.
        output_dir: Directory for index.html and the overlay PNG. Created if
            missing.
        deposit_file: Optional deposit vector to overlay as yellow markers.
        fault_file: Optional fault vector to overlay as red lines (thrust dashed).
        commodity_filter: Keep only deposits whose commodity attribute contains
            this text (e.g. "Au").
        fallback_crs: CRS assigned to vectors lacking a defined CRS.
        title: Title shown on the map.

    Returns:
        Filepath of the written index.html.
    """
    import folium

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    array, (south, west, north, east) = _reproject_to_wgs84(prospectivity_raster)
    png_path = output_dir / "prospectivity_overlay.png"
    _array_to_png(array, png_path)

    # Embed the overlay as a data URI so index.html is self-contained and works
    # on GitHub Pages from a relative URL without a separate fetch.
    import base64

    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    overlay_uri = f"data:image/png;base64,{encoded}"

    center = [(south + north) / 2, (west + east) / 2]
    fmap = folium.Map(location=center, zoom_start=9, tiles="OpenStreetMap", control_scale=True)
    folium.TileLayer("CartoDB positron", name="Light basemap").add_to(fmap)

    folium.raster_layers.ImageOverlay(
        name="Gold prospectivity",
        image=overlay_uri,
        bounds=[[south, west], [north, east]],
        opacity=0.75,
    ).add_to(fmap)

    def _load(path):
        gdf = gpd.read_file(path)
        if gdf.crs is None and fallback_crs:
            gdf = gdf.set_crs(fallback_crs)
        return gdf.to_crs(WEB_CRS) if gdf.crs and str(gdf.crs) != WEB_CRS else gdf

    # Faults: thrust dashed, others solid, both red.
    if fault_file:
        faults = _load(fault_file)
        thrust, other = _split_faults(faults)
        other_lines = _fault_lines(other)
        thrust_lines = _fault_lines(thrust)
        if not other_lines.empty:
            folium.GeoJson(
                other_lines.to_json(),
                name="Faults",
                style_function=lambda _: {"color": "red", "weight": 1.5},
            ).add_to(fmap)
        if not thrust_lines.empty:
            folium.GeoJson(
                thrust_lines.to_json(),
                name="Thrust faults",
                style_function=lambda _: {"color": "red", "weight": 2, "dashArray": "6, 6"},
            ).add_to(fmap)

    # Deposits: yellow markers.
    if deposit_file:
        deposits = _load(deposit_file)
        if commodity_filter:
            cols = [c for c in deposits.columns if "COMMODITY" in c.upper()]
            if cols:
                mask = np.zeros(len(deposits), dtype=bool)
                for col in cols:
                    mask |= deposits[col].astype(str).str.contains(commodity_filter, case=False, na=False)
                deposits = deposits[mask]
        points = deposits[deposits.geometry.type.isin(["Point", "MultiPoint"])]
        # One GeoJson layer with a shared CircleMarker keeps the file small even
        # with thousands of deposits (vs. one marker object each).
        folium.GeoJson(
            points[[points.geometry.name]].to_json(),
            name="Deposits",
            marker=folium.CircleMarker(
                radius=3, color="black", weight=0.5, fill=True,
                fill_color="yellow", fill_opacity=0.9,
            ),
        ).add_to(fmap)

    title_html = (
        f'<div style="position: fixed; top: 10px; left: 50px; z-index: 9999; '
        f'background: white; padding: 6px 12px; border: 1px solid #888; '
        f'border-radius: 4px; font-family: sans-serif; font-weight: bold;">{title}</div>'
    )
    fmap.get_root().html.add_child(folium.Element(title_html))
    folium.LayerControl(collapsed=False).add_to(fmap)

    index_path = output_dir / "index.html"
    fmap.save(str(index_path))
    return str(index_path)
