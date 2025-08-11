"""Configure and extract sampling data from OpenFOAM."""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from amx.config import Config


class SamplingConfig:
    """Configure OpenFOAM sampling functionObjects."""

    def __init__(self, config: Config):
        """
        Initialize sampling configuration.
        
        Args:
            config: Simulation configuration
        """
        self.config = config

    def create_plane_sampling_dict(self) -> str:
        """Create dictionary entries for plane sampling."""
        if not self.config.export or not self.config.export.planes:
            return ""
        
        entries = []
        for i, plane in enumerate(self.config.export.planes):
            z = plane.get("z", 0)
            entry = f"""
    planeSample{i}
    {{
        type            surfaces;
        libs            ("libsampling.so");
        writeControl    timeStep;
        writeInterval   {self.config.export.sample_interval};
        surfaceFormat   vtk;
        fields          ({' '.join(self.config.export.fields)});
        interpolationScheme cellPoint;
        surfaces
        (
            plane{i}
            {{
                type        cuttingPlane;
                planeType   pointAndNormal;
                pointAndNormalDict
                {{
                    point   (0 0 {z});
                    normal  (0 0 1);
                }}
                interpolate true;
            }}
        );
    }}"""
            entries.append(entry)
        
        return "\n".join(entries)

    def create_line_sampling_dict(self, lines: List[Dict]) -> str:
        """
        Create dictionary entries for line sampling.
        
        Args:
            lines: List of line definitions with start/end points
        """
        entries = []
        for i, line in enumerate(lines):
            start = line["start"]
            end = line["end"]
            entry = f"""
    lineSample{i}
    {{
        type            sets;
        libs            ("libsampling.so");
        writeControl    timeStep;
        writeInterval   {self.config.export.sample_interval};
        setFormat       csv;
        fields          ({' '.join(self.config.export.fields)});
        interpolationScheme cellPoint;
        sets
        (
            line{i}
            {{
                type    uniform;
                axis    distance;
                start   ({start[0]} {start[1]} {start[2]});
                end     ({end[0]} {end[1]} {end[2]});
                nPoints 100;
            }}
        );
    }}"""
            entries.append(entry)
        
        return "\n".join(entries)

    def create_probe_sampling_dict(self, points: List[List[float]]) -> str:
        """
        Create dictionary entries for probe sampling.
        
        Args:
            points: List of probe point coordinates
        """
        probe_points = "\n        ".join(
            [f"({p[0]} {p[1]} {p[2]})" for p in points]
        )
        
        entry = f"""
    probes
    {{
        type            probes;
        libs            ("libsampling.so");
        writeControl    timeStep;
        writeInterval   1;
        fields          ({' '.join(self.config.export.fields)});
        probeLocations
        (
            {probe_points}
        );
    }}"""
        
        return entry


class SamplingExtractor:
    """Extract and process sampling data from OpenFOAM results."""

    def __init__(self, case_dir: Path):
        """
        Initialize sampling extractor.
        
        Args:
            case_dir: Case directory path
        """
        self.case_dir = Path(case_dir)
        self.postprocessing_dir = self.case_dir / "postProcessing"

    def extract_plane_data(self, time: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """
        Extract plane sampling data.
        
        Args:
            time: Time to extract (None for latest)
            
        Returns:
            Dictionary of DataFrames for each plane
        """
        plane_data = {}
        
        # Find plane sampling directories
        for item in self.postprocessing_dir.iterdir():
            if item.is_dir() and "planeSample" in item.name:
                plane_name = item.name
                
                # Get time directory
                time_dir = self._get_time_dir(item, time)
                if not time_dir:
                    continue
                
                # Read VTK files
                vtk_files = list(time_dir.glob("*.vtk"))
                if vtk_files:
                    # Would need VTK reader here
                    # For now, store file paths
                    plane_data[plane_name] = str(vtk_files[0])
        
        return plane_data

    def extract_line_data(self, time: Optional[float] = None) -> Dict[str, pd.DataFrame]:
        """
        Extract line sampling data.
        
        Args:
            time: Time to extract (None for latest)
            
        Returns:
            Dictionary of DataFrames for each line
        """
        line_data = {}
        
        # Find line sampling directories
        for item in self.postprocessing_dir.iterdir():
            if item.is_dir() and "lineSample" in item.name:
                line_name = item.name
                
                # Get time directory
                time_dir = self._get_time_dir(item, time)
                if not time_dir:
                    continue
                
                # Read CSV files
                csv_files = list(time_dir.glob("*.csv"))
                for csv_file in csv_files:
                    df = pd.read_csv(csv_file)
                    line_data[f"{line_name}_{csv_file.stem}"] = df
        
        return line_data

    def extract_probe_data(self) -> pd.DataFrame:
        """
        Extract probe data time series.
        
        Returns:
            DataFrame with probe data
        """
        probes_dir = self.postprocessing_dir / "probes" / "0"
        if not probes_dir.exists():
            return pd.DataFrame()
        
        # Read probe data files
        data_files = list(probes_dir.glob("*.dat"))
        if not data_files:
            return pd.DataFrame()
        
        # Combine all probe data
        all_data = []
        for data_file in data_files:
            field_name = data_file.stem
            df = pd.read_csv(data_file, sep=r'\s+', comment='#', header=None)
            df.columns = ['time'] + [f'{field_name}_{i}' for i in range(len(df.columns)-1)]
            all_data.append(df)
        
        # Merge on time column
        if all_data:
            result = all_data[0]
            for df in all_data[1:]:
                result = pd.merge(result, df, on='time', how='outer')
            return result
        
        return pd.DataFrame()

    def _get_time_dir(self, parent_dir: Path, time: Optional[float] = None) -> Optional[Path]:
        """Get time directory (latest if time is None)."""
        time_dirs = []
        for item in parent_dir.iterdir():
            if item.is_dir() and item.name.replace(".", "").isdigit():
                try:
                    time_val = float(item.name)
                    time_dirs.append((time_val, item))
                except ValueError:
                    pass
        
        if not time_dirs:
            return None
        
        time_dirs.sort(key=lambda x: x[0])
        
        if time is None:
            return time_dirs[-1][1]
        else:
            # Find closest time
            for t, path in time_dirs:
                if abs(t - time) < 0.01:
                    return path
            return None