"""Mesh generation for OpenFOAM cases."""

import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from jinja2 import Template

from amx.config import Config
from amx.geometry import NozzleArray, TankGeometry


BLOCKMESH_TEMPLATE = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v11                                   |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      blockMeshDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

convertToMeters 1;

vertices
(
    (0 0 0)           // 0
    ({{ L }} 0 0)     // 1
    ({{ L }} {{ W }} 0)  // 2
    (0 {{ W }} 0)     // 3
    (0 0 {{ H }})     // 4
    ({{ L }} 0 {{ H }})  // 5
    ({{ L }} {{ W }} {{ H }})  // 6
    (0 {{ W }} {{ H }})  // 7
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({{ nx }} {{ ny }} {{ nz }}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    walls
    {
        type wall;
        faces
        (
            (0 4 7 3)  // left
            (1 2 6 5)  // right
            (0 1 5 4)  // front
            (3 7 6 2)  // back
        );
    }
    
    bottom
    {
        type wall;
        faces
        (
            (0 3 2 1)
        );
    }
    
    top
    {
        type patch;
        faces
        (
            (4 5 6 7)
        );
    }
);

mergePatchPairs
(
);

// ************************************************************************* //
"""

TOPOSET_TEMPLATE = """/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  v11                                   |
|   \\\\  /    A nd           | Web:      www.OpenFOAM.org                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      topoSetDict;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

actions
(
{% for zone in nozzle_zones %}
    {
        name    nozzleZone_{{ zone.id }};
        type    cellZoneSet;
        action  new;
        source  cylinderToCell;
        sourceInfo
        {
            p1      ({{ zone.p1[0] }} {{ zone.p1[1] }} {{ zone.p1[2] }});
            p2      ({{ zone.p2[0] }} {{ zone.p2[1] }} {{ zone.p2[2] }});
            radius  {{ zone.radius }};
        }
    }
{% endfor %}

{% if screen_zone %}
    {
        name    screenZone;
        type    cellZoneSet;
        action  new;
        source  cylinderToCell;
        sourceInfo
        {
            p1      ({{ screen_zone.p1[0] }} {{ screen_zone.p1[1] }} {{ screen_zone.p1[2] }});
            p2      ({{ screen_zone.p2[0] }} {{ screen_zone.p2[1] }} {{ screen_zone.p2[2] }});
            radius  {{ screen_zone.radius }};
        }
    }
{% endif %}
);

// ************************************************************************* //
"""


class MeshGenerator:
    """Generate mesh for OpenFOAM simulations."""

    def __init__(self, config: Config):
        """
        Initialize mesh generator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.blockmesh_template = Template(BLOCKMESH_TEMPLATE)
        self.toposet_template = Template(TOPOSET_TEMPLATE)

    def calculate_cell_counts(self) -> Tuple[int, int, int]:
        """Calculate number of cells in each direction."""
        L = self.config.geometry.tank.L
        W = self.config.geometry.tank.W
        H = self.config.geometry.tank.H
        
        cell_size = 0.35  # Default
        if self.config.geometry.mesh:
            cell_size = self.config.geometry.mesh.base_cell_size
        
        nx = max(int(L / cell_size), 10)
        ny = max(int(W / cell_size), 10)
        nz = max(int(H / cell_size), 10)
        
        return (nx, ny, nz)

    def write_blockmesh_dict(self, case_dir: Path) -> None:
        """Write blockMeshDict file."""
        nx, ny, nz = self.calculate_cell_counts()
        
        content = self.blockmesh_template.render(
            L=self.config.geometry.tank.L,
            W=self.config.geometry.tank.W,
            H=self.config.geometry.tank.H,
            nx=nx,
            ny=ny,
            nz=nz,
        )
        
        output_path = case_dir / "system" / "blockMeshDict"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    def write_toposet_dict(self, case_dir: Path) -> None:
        """Write topoSetDict file for cell zones."""
        nozzle_zones = []
        screen_zone = None
        
        # Generate nozzle zones
        if self.config.geometry.nozzle:
            tank = TankGeometry(
                self.config.geometry.tank.L,
                self.config.geometry.tank.W,
                self.config.geometry.tank.H,
            )
            
            nozzle_array = NozzleArray(
                count=self.config.geometry.nozzle.count,
                array=self.config.geometry.nozzle.array,
                tank=tank,
                pitch_deg=self.config.geometry.nozzle.pitch_deg,
                start_offset=self.config.geometry.nozzle.start_offset_m,
                spacing=self.config.geometry.nozzle.spacing_m,
            )
            
            zones = nozzle_array.get_momentum_zones(
                radius=self.config.geometry.nozzle.zone_radius_m,
                height=self.config.geometry.nozzle.zone_height_m,
            )
            
            for i, zone in enumerate(zones):
                # Calculate cylinder endpoints
                center = zone["center"]
                direction = zone["direction"]
                height = zone["height"]
                
                p1 = [
                    center[0] - direction[0] * height / 2,
                    center[1] - direction[1] * height / 2,
                    center[2] - direction[2] * height / 2,
                ]
                p2 = [
                    center[0] + direction[0] * height / 2,
                    center[1] + direction[1] * height / 2,
                    center[2] + direction[2] * height / 2,
                ]
                
                nozzle_zones.append({
                    "id": f"{i:02d}",
                    "p1": p1,
                    "p2": p2,
                    "radius": zone["radius"],
                })
        
        # Generate screen zone
        if self.config.geometry.screen:
            center = self.config.geometry.screen.center
            height = self.config.geometry.screen.height
            
            screen_zone = {
                "p1": [center[0], center[1], center[2] - height / 2],
                "p2": [center[0], center[1], center[2] + height / 2],
                "radius": self.config.geometry.screen.radius,
            }
        
        content = self.toposet_template.render(
            nozzle_zones=nozzle_zones,
            screen_zone=screen_zone,
        )
        
        output_path = case_dir / "system" / "topoSetDict"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)

    def generate_mesh(self, case_dir: Path) -> bool:
        """
        Generate mesh using blockMesh and topoSet.
        
        Args:
            case_dir: Case directory path
            
        Returns:
            True if successful, False otherwise
        """
        # Write mesh dictionaries
        self.write_blockmesh_dict(case_dir)
        self.write_toposet_dict(case_dir)
        
        # Run blockMesh
        try:
            result = subprocess.run(
                ["blockMesh", "-case", str(case_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"blockMesh completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"blockMesh failed: {e.stderr}")
            return False
        
        # Run topoSet
        try:
            result = subprocess.run(
                ["topoSet", "-case", str(case_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"topoSet completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"topoSet failed: {e.stderr}")
            return False
        
        return True