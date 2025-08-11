"""I/O utilities for reading VTK and CSV files."""

from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import pyvista as pv


class VTKReader:
    """Read VTK files from OpenFOAM output."""

    def __init__(self, case_dir: Path):
        """
        Initialize VTK reader.
        
        Args:
            case_dir: OpenFOAM case directory
        """
        self.case_dir = Path(case_dir)
        self.vtk_dir = self.case_dir / "VTK"

    def read_time_step(self, time: Optional[float] = None) -> pv.UnstructuredGrid:
        """
        Read VTK data for a specific time step.
        
        Args:
            time: Time value (None for latest)
            
        Returns:
            PyVista UnstructuredGrid
        """
        # Find VTK files
        vtk_files = self._find_vtk_files(time)
        if not vtk_files:
            raise FileNotFoundError(f"No VTK files found for time {time}")
        
        # Read the internal mesh
        internal_file = None
        for f in vtk_files:
            if "internalMesh" in f.name:
                internal_file = f
                break
        
        if not internal_file:
            # Use first available file
            internal_file = vtk_files[0]
        
        # Read with PyVista
        mesh = pv.read(str(internal_file))
        
        return mesh

    def read_all_times(self) -> Dict[float, pv.UnstructuredGrid]:
        """
        Read all available time steps.
        
        Returns:
            Dictionary mapping time to mesh
        """
        times = self.get_available_times()
        meshes = {}
        
        for t in times:
            try:
                meshes[t] = self.read_time_step(t)
            except Exception as e:
                print(f"Warning: Could not read time {t}: {e}")
        
        return meshes

    def get_available_times(self) -> List[float]:
        """Get list of available time values."""
        times = []
        
        if not self.vtk_dir.exists():
            return times
        
        # Look for time directories
        for item in self.vtk_dir.iterdir():
            if item.is_dir():
                try:
                    time_val = float(item.name)
                    times.append(time_val)
                except ValueError:
                    pass
        
        return sorted(times)

    def _find_vtk_files(self, time: Optional[float] = None) -> List[Path]:
        """Find VTK files for given time."""
        if time is None:
            # Get latest time
            times = self.get_available_times()
            if not times:
                return []
            time = times[-1]
        
        time_dir = self.vtk_dir / str(time)
        if not time_dir.exists():
            # Try with different precision
            for t_dir in self.vtk_dir.iterdir():
                if t_dir.is_dir():
                    try:
                        t_val = float(t_dir.name)
                        if abs(t_val - time) < 0.01:
                            time_dir = t_dir
                            break
                    except ValueError:
                        pass
        
        if not time_dir.exists():
            return []
        
        return list(time_dir.glob("*.vtk"))


class CSVReader:
    """Read CSV files from OpenFOAM sampling."""

    def __init__(self, case_dir: Path):
        """
        Initialize CSV reader.
        
        Args:
            case_dir: OpenFOAM case directory
        """
        self.case_dir = Path(case_dir)
        self.postprocessing_dir = self.case_dir / "postProcessing"

    def read_line_data(self, line_name: str, time: Optional[float] = None) -> pd.DataFrame:
        """
        Read line sampling data.
        
        Args:
            line_name: Name of the sampled line
            time: Time value (None for latest)
            
        Returns:
            DataFrame with line data
        """
        line_dir = self.postprocessing_dir / line_name
        if not line_dir.exists():
            raise FileNotFoundError(f"Line data directory not found: {line_dir}")
        
        # Find time directory
        time_dir = self._find_time_dir(line_dir, time)
        if not time_dir:
            raise FileNotFoundError(f"No data found for time {time}")
        
        # Read CSV files
        csv_files = list(time_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No CSV files in {time_dir}")
        
        # Read and combine data
        data_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file)
            data_frames.append(df)
        
        if len(data_frames) == 1:
            return data_frames[0]
        else:
            # Merge multiple files
            result = data_frames[0]
            for df in data_frames[1:]:
                # Merge on position columns
                merge_cols = [c for c in df.columns if c in ['x', 'y', 'z', 'distance']]
                if merge_cols:
                    result = pd.merge(result, df, on=merge_cols, how='outer')
            return result

    def read_probe_data(self, probe_name: str = "probes") -> pd.DataFrame:
        """
        Read probe time series data.
        
        Args:
            probe_name: Name of the probe set
            
        Returns:
            DataFrame with time series data
        """
        probe_dir = self.postprocessing_dir / probe_name / "0"
        if not probe_dir.exists():
            raise FileNotFoundError(f"Probe data directory not found: {probe_dir}")
        
        # Read all data files
        data_files = list(probe_dir.glob("*.dat"))
        if not data_files:
            raise FileNotFoundError(f"No probe data files in {probe_dir}")
        
        # Read and combine
        all_data = {}
        for data_file in data_files:
            field_name = data_file.stem
            
            # Read with proper handling of OpenFOAM format
            with open(data_file, 'r') as f:
                lines = f.readlines()
            
            # Skip header lines
            data_lines = [l for l in lines if not l.startswith('#')]
            
            # Parse data
            data = []
            for line in data_lines:
                parts = line.strip().split()
                if parts:
                    data.append([float(x) for x in parts])
            
            if data:
                df = pd.DataFrame(data)
                # First column is time
                df.columns = ['time'] + [f'{field_name}_probe_{i}' for i in range(len(df.columns)-1)]
                
                if 'time' not in all_data:
                    all_data['time'] = df['time']
                
                for col in df.columns[1:]:
                    all_data[col] = df[col]
        
        return pd.DataFrame(all_data)

    def _find_time_dir(self, parent_dir: Path, time: Optional[float] = None) -> Optional[Path]:
        """Find time directory."""
        time_dirs = []
        
        for item in parent_dir.iterdir():
            if item.is_dir():
                try:
                    t_val = float(item.name)
                    time_dirs.append((t_val, item))
                except ValueError:
                    pass
        
        if not time_dirs:
            return None
        
        time_dirs.sort(key=lambda x: x[0])
        
        if time is None:
            # Return latest
            return time_dirs[-1][1]
        else:
            # Find closest
            for t_val, t_dir in time_dirs:
                if abs(t_val - time) < 0.01:
                    return t_dir
            return None


class DataExporter:
    """Export processed data to various formats."""

    @staticmethod
    def to_csv(data: Union[pd.DataFrame, Dict], output_path: Path) -> None:
        """
        Export data to CSV.
        
        Args:
            data: DataFrame or dict of DataFrames
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_csv(output_path, index=False)
        elif isinstance(data, dict):
            # Save each item as separate CSV
            for key, df in data.items():
                if isinstance(df, pd.DataFrame):
                    file_path = output_path.parent / f"{output_path.stem}_{key}.csv"
                    df.to_csv(file_path, index=False)

    @staticmethod
    def to_excel(data: Union[pd.DataFrame, Dict], output_path: Path) -> None:
        """
        Export data to Excel.
        
        Args:
            data: DataFrame or dict of DataFrames
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_excel(output_path, index=False)
        elif isinstance(data, dict):
            # Save each item as separate sheet
            with pd.ExcelWriter(output_path) as writer:
                for key, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        # Limit sheet name to 31 characters (Excel limit)
                        sheet_name = key[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def to_json(data: Union[pd.DataFrame, Dict], output_path: Path) -> None:
        """
        Export data to JSON.
        
        Args:
            data: DataFrame or dict
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if isinstance(data, pd.DataFrame):
            data.to_json(output_path, orient='records', indent=2)
        elif isinstance(data, dict):
            import json
            # Convert DataFrames to dict
            json_data = {}
            for key, value in data.items():
                if isinstance(value, pd.DataFrame):
                    json_data[key] = value.to_dict('records')
                else:
                    json_data[key] = value
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2, default=str)