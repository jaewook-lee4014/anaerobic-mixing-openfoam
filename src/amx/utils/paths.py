"""Path and directory utilities."""

from pathlib import Path
from typing import List, Optional


def create_case_structure(case_dir: Path) -> None:
    """
    Create OpenFOAM case directory structure.
    
    Args:
        case_dir: Case directory path
    """
    case_dir = Path(case_dir)
    
    # Create main directories
    dirs = [
        case_dir / "0",
        case_dir / "constant",
        case_dir / "system",
    ]
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def get_latest_time(case_dir: Path) -> Optional[float]:
    """
    Get latest time directory from case.
    
    Args:
        case_dir: Case directory path
        
    Returns:
        Latest time value or None
    """
    case_dir = Path(case_dir)
    
    time_dirs = []
    for item in case_dir.iterdir():
        if item.is_dir() and item.name.replace(".", "").replace("-", "").isdigit():
            try:
                time_val = float(item.name)
                time_dirs.append(time_val)
            except ValueError:
                pass
    
    if time_dirs:
        return max(time_dirs)
    return None


def find_vtk_files(case_dir: Path, time: Optional[float] = None) -> List[Path]:
    """
    Find VTK files in case directory.
    
    Args:
        case_dir: Case directory path
        time: Specific time (latest if None)
        
    Returns:
        List of VTK file paths
    """
    case_dir = Path(case_dir)
    vtk_dir = case_dir / "VTK"
    
    if not vtk_dir.exists():
        return []
    
    if time is None:
        # Find latest time directory
        time_dirs = []
        for item in vtk_dir.iterdir():
            if item.is_dir():
                try:
                    t = float(item.name)
                    time_dirs.append((t, item))
                except ValueError:
                    pass
        
        if not time_dirs:
            return []
        
        time_dirs.sort(key=lambda x: x[0])
        time_dir = time_dirs[-1][1]
    else:
        time_dir = vtk_dir / str(time)
        if not time_dir.exists():
            return []
    
    return list(time_dir.glob("*.vtk"))


def clean_case_dir(case_dir: Path, keep_latest: bool = True) -> None:
    """
    Clean case directory of result files.
    
    Args:
        case_dir: Case directory path
        keep_latest: Keep latest time directory
    """
    case_dir = Path(case_dir)
    
    # Find time directories
    time_dirs = []
    for item in case_dir.iterdir():
        if item.is_dir() and item.name.replace(".", "").isdigit():
            try:
                time_val = float(item.name)
                if time_val > 0:  # Don't delete 0 directory
                    time_dirs.append((time_val, item))
            except ValueError:
                pass
    
    if keep_latest and time_dirs:
        # Sort and remove latest
        time_dirs.sort(key=lambda x: x[0])
        time_dirs = time_dirs[:-1]
    
    # Delete directories
    for _, dir_path in time_dirs:
        import shutil
        shutil.rmtree(dir_path)
    
    # Clean processor directories
    for item in case_dir.glob("processor*"):
        if item.is_dir():
            import shutil
            shutil.rmtree(item)
    
    # Clean postProcessing
    post_dir = case_dir / "postProcessing"
    if post_dir.exists():
        import shutil
        shutil.rmtree(post_dir)
    
    # Clean VTK
    vtk_dir = case_dir / "VTK"
    if vtk_dir.exists():
        import shutil
        shutil.rmtree(vtk_dir)