"""Minimal desktop GUI for the mineral prospectivity mapping workflow.

Provides a single window with file pickers for the four required inputs (DEM,
fault, geology, deposits), optional extra evidence rasters, a model selector and
an output folder, plus a single "Run" button. The workflow runs on a background
thread so the window stays responsive, and progress/results are shown in a log
pane.

Launch with::

    eis-mpm-gui

or::

    python -m eis_toolkit.workflows.gui
"""

import os
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

MODELS = ["random_forest", "gradient_boosting", "logistic_regression"]


class MpmApp:
    """Tkinter application wrapping the mineral prospectivity workflow."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("EIS Toolkit - Mineral Prospectivity Mapping")
        root.geometry("760x620")

        self.vars = {
            "dem": tk.StringVar(),
            "fault": tk.StringVar(),
            "geology": tk.StringVar(),
            "deposits": tk.StringVar(),
            "output_dir": tk.StringVar(),
            "commodity_filter": tk.StringVar(value="Au"),
            "fallback_crs": tk.StringVar(),
            "n_estimators": tk.StringVar(value="100"),
        }
        self.model_var = tk.StringVar(value=MODELS[0])
        self.compare_var = tk.BooleanVar(value=False)
        self.extra_rasters: list = []
        self._log_queue: "queue.Queue[str]" = queue.Queue()
        self._result_queue: "queue.Queue[dict]" = queue.Queue()
        self._canvas = None

        self._build_widgets()
        self._poll_log()

    # ----- UI construction -------------------------------------------------

    def _build_widgets(self) -> None:
        pad = {"padx": 6, "pady": 4}
        frame = ttk.Frame(self.root)
        frame.pack(fill="x", **pad)

        self._file_row(frame, 0, "DEM raster", "dem", self._pick_file)
        self._file_row(frame, 1, "Fault vector", "fault", self._pick_file)
        self._file_row(frame, 2, "Geology vector", "geology", self._pick_file)
        self._file_row(frame, 3, "Deposits vector", "deposits", self._pick_file)
        self._file_row(frame, 4, "Output folder", "output_dir", self._pick_dir)

        # Extra rasters
        ttk.Label(frame, text="Extra rasters").grid(row=5, column=0, sticky="w", **pad)
        self.extra_label = ttk.Label(frame, text="(none)", foreground="gray")
        self.extra_label.grid(row=5, column=1, sticky="w", **pad)
        ttk.Button(frame, text="Add...", command=self._add_extra).grid(row=5, column=2, **pad)
        ttk.Button(frame, text="Clear", command=self._clear_extra).grid(row=5, column=3, **pad)

        # Options
        opts = ttk.LabelFrame(self.root, text="Options")
        opts.pack(fill="x", **pad)
        ttk.Label(opts, text="Model").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(opts, textvariable=self.model_var, values=MODELS, state="readonly", width=22).grid(
            row=0, column=1, sticky="w", **pad
        )
        ttk.Label(opts, text="Commodity filter").grid(row=0, column=2, sticky="w", **pad)
        ttk.Entry(opts, textvariable=self.vars["commodity_filter"], width=10).grid(row=0, column=3, **pad)
        ttk.Checkbutton(opts, text="Compare all models", variable=self.compare_var).grid(
            row=0, column=4, sticky="w", **pad
        )
        ttk.Label(opts, text="Fallback CRS").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(opts, textvariable=self.vars["fallback_crs"], width=14).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(opts, text="n_estimators").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(opts, textvariable=self.vars["n_estimators"], width=10).grid(row=1, column=3, **pad)

        # Run button + progress
        run_frame = ttk.Frame(self.root)
        run_frame.pack(fill="x", **pad)
        self.run_button = ttk.Button(run_frame, text="Run", command=self._on_run)
        self.run_button.pack(side="left", padx=6)
        self.progress = ttk.Progressbar(run_frame, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=6)

        # Tabbed output: Log + Map preview
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, **pad)

        log_frame = ttk.Frame(notebook)
        notebook.add(log_frame, text="Log")
        self.log = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log.pack(side="left", fill="both", expand=True)
        scroll = ttk.Scrollbar(log_frame, command=self.log.yview)
        scroll.pack(side="right", fill="y")
        self.log.configure(yscrollcommand=scroll.set)

        self.preview_frame = ttk.Frame(notebook)
        notebook.add(self.preview_frame, text="Map preview")
        self.notebook = notebook

    def _file_row(self, frame, row, label, key, picker):
        pad = {"padx": 6, "pady": 4}
        ttk.Label(frame, text=label).grid(row=row, column=0, sticky="w", **pad)
        ttk.Entry(frame, textvariable=self.vars[key], width=58).grid(row=row, column=1, columnspan=2, sticky="we", **pad)
        ttk.Button(frame, text="Browse...", command=lambda: picker(key)).grid(row=row, column=3, **pad)
        frame.columnconfigure(1, weight=1)

    # ----- Pickers ---------------------------------------------------------

    def _pick_file(self, key):
        path = filedialog.askopenfilename(title=f"Select {key}")
        if path:
            self.vars[key].set(path)

    def _pick_dir(self, key):
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.vars[key].set(path)

    def _add_extra(self):
        paths = filedialog.askopenfilenames(title="Select extra rasters")
        self.extra_rasters.extend(paths)
        self._refresh_extra()

    def _clear_extra(self):
        self.extra_rasters = []
        self._refresh_extra()

    def _refresh_extra(self):
        if self.extra_rasters:
            names = ", ".join(os.path.basename(p) for p in self.extra_rasters)
            self.extra_label.configure(text=names, foreground="black")
        else:
            self.extra_label.configure(text="(none)", foreground="gray")

    # ----- Run -------------------------------------------------------------

    def _log_message(self, message: str) -> None:
        self._log_queue.put(message)

    def _poll_log(self) -> None:
        while not self._log_queue.empty():
            message = self._log_queue.get_nowait()
            self.log.configure(state="normal")
            self.log.insert("end", message + "\n")
            self.log.see("end")
            self.log.configure(state="disabled")
        self.root.after(150, self._poll_log)

    def _on_run(self) -> None:
        required = ["dem", "fault", "geology", "deposits", "output_dir"]
        missing = [key for key in required if not self.vars[key].get()]
        if missing:
            messagebox.showerror("Missing inputs", f"Please provide: {', '.join(missing)}")
            return
        try:
            n_estimators = int(self.vars["n_estimators"].get())
        except ValueError:
            messagebox.showerror("Invalid value", "n_estimators must be an integer.")
            return

        self.run_button.configure(state="disabled")
        self.progress.start(12)
        self._log_message("Starting workflow...")

        args = dict(
            dem_file=self.vars["dem"].get(),
            fault_file=self.vars["fault"].get(),
            geology_file=self.vars["geology"].get(),
            deposit_file=self.vars["deposits"].get(),
            output_dir=self.vars["output_dir"].get(),
            extra_rasters=list(self.extra_rasters) or None,
            model=self.model_var.get(),
            compare=self.compare_var.get(),
            commodity_filter=self.vars["commodity_filter"].get() or None,
            fallback_crs=self.vars["fallback_crs"].get() or None,
            n_estimators=n_estimators,
        )
        threading.Thread(target=self._worker, args=(args,), daemon=True).start()

    def _worker(self, args: dict) -> None:
        try:
            os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")
            from eis_toolkit.workflows.mineral_prospectivity import run_mineral_prospectivity_workflow

            outputs = run_mineral_prospectivity_workflow(**args)
            self._log_message("Workflow finished. Outputs:")
            for name, path in outputs.items():
                self._log_message(f"  {name}: {path}")
            self._result_queue.put(
                {"raster": outputs.get("prospectivity_raster"), "deposits": args["deposit_file"],
                 "commodity": args.get("commodity_filter"), "fallback_crs": args.get("fallback_crs")}
            )
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user
            self._log_message(f"ERROR: {exc}")
            self.root.after(0, lambda: messagebox.showerror("Workflow failed", str(exc)))
        finally:
            self.root.after(0, self._finish)

    def _finish(self) -> None:
        self.progress.stop()
        self.run_button.configure(state="normal")
        while not self._result_queue.empty():
            self._show_preview(self._result_queue.get_nowait())

    def _show_preview(self, result: dict) -> None:
        """Render the resulting prospectivity raster into the preview tab."""
        raster_path = result.get("raster")
        if not raster_path or not os.path.exists(raster_path):
            return
        try:
            import numpy as np
            import rasterio
            from rasterio.transform import array_bounds
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

            with rasterio.open(raster_path) as src:
                data = src.read(1)
                profile = src.profile
            masked = np.ma.masked_equal(data, profile.get("nodata", -9999.0))
            height, width = data.shape
            left, bottom, right, top = array_bounds(height, width, profile["transform"])

            for child in self.preview_frame.winfo_children():
                child.destroy()

            figure = Figure(figsize=(6, 5))
            axis = figure.add_subplot(111)
            image = axis.imshow(masked, cmap="magma", extent=(left, right, bottom, top), vmin=0, vmax=1)
            figure.colorbar(image, ax=axis, shrink=0.7, label="Prospectivity")
            axis.set_title("Prospectivity map")
            axis.set_xlabel("Easting")
            axis.set_ylabel("Northing")

            self._canvas = FigureCanvasTkAgg(figure, master=self.preview_frame)
            self._canvas.draw()
            self._canvas.get_tk_widget().pack(fill="both", expand=True)
            self.notebook.select(self.preview_frame)
        except Exception as exc:  # noqa: BLE001
            self._log_message(f"Preview unavailable: {exc}")


def main() -> None:
    """Launch the mineral prospectivity GUI."""
    root = tk.Tk()
    MpmApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
