"""Generate fvOptions for momentum sources and porous zones."""

from pathlib import Path
from typing import Dict, List, Optional

from jinja2 import Template

from amx.config import Config
from amx.geometry import NozzleArray, ScreenGeometry, TankGeometry


FVOPTIONS_TEMPLATE = """/*--------------------------------*- C++ -*----------------------------------*\\
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
    location    "constant";
    object      fvOptions;
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

{
{% for nozzle in nozzles %}
    nozzleSource_{{ nozzle.id }}
    {
        type            scalarCodedSource;
        active          yes;
        name            nozzle{{ nozzle.id }};
        
        scalarCodedSourceCoeffs
        {
            selectionMode   cellZone;
            cellZone        nozzleZone_{{ nozzle.id }};
            
            fields          (U);
            name            nozzle{{ nozzle.id }};
            
            codeInclude
            #{
                #include "fvCFD.H"
            #};
            
            codeCorrect
            #{
            #};
            
            codeAddSup
            #{
                const Time& time = mesh_.time();
                const vectorField& C = mesh_.C();
                const scalarField& V = mesh_.V();
                vectorField& USource = eqn.source();
                
                const labelList& cells = mesh_.cellZones()[cellZoneID_];
                
                // Jet parameters
                const vector nozzlePos({{ nozzle.center[0] }}, {{ nozzle.center[1] }}, {{ nozzle.center[2] }});
                const vector dir({{ nozzle.direction[0] }}, {{ nozzle.direction[1] }}, {{ nozzle.direction[2] }});
                const scalar flowRate = {{ nozzle.flow_rate }};  // m3/s
                const scalar rho = {{ rho }};  // kg/m3
                const scalar nozzleDiam = {{ nozzle.diameter }};  // m
                const scalar jetVelocity = flowRate / (3.14159 * pow(nozzleDiam/2, 2));
                
                // Calculate zone volume
                scalar zoneVolume = 0.0;
                forAll(cells, i)
                {
                    zoneVolume += V[cells[i]];
                }
                
                if (zoneVolume > SMALL)
                {
                    // Momentum flux from jet
                    const scalar momentumFlux = rho * flowRate * jetVelocity;
                    
                    // Distribute momentum with spatial weighting
                    forAll(cells, i)
                    {
                        label cellI = cells[i];
                        const vector& cellCenter = C[cellI];
                        
                        // Distance from nozzle center
                        scalar dist = mag(cellCenter - nozzlePos);
                        
                        // Gaussian distribution of momentum source
                        scalar sigma = nozzleDiam * 2.0;  // Spread parameter
                        scalar weight = exp(-pow(dist/sigma, 2));
                        
                        // Apply momentum source with weighting
                        scalar localMomentum = momentumFlux * weight / zoneVolume;
                        USource[cellI] += localMomentum * dir * V[cellI];
                    }
                    
                    // Add turbulence generation (simplified)
                    // This enhances mixing by increasing local turbulence
                    // Actual implementation would modify k and epsilon equations
                }
            #};
            
            codeSetValue
            #{
            #};
        }
    }
{% endfor %}

{% if screen %}
    screenPorousZone
    {
        type            explicitPorositySource;
        active          yes;
        
        explicitPorositySourceCoeffs
        {
            selectionMode   cellZone;
            cellZone        screenZone;
            
            type            DarcyForchheimer;
            
            DarcyForchheimerCoeffs
            {
                d   ({{ screen.darcy[0] }} {{ screen.darcy[1] }} {{ screen.darcy[2] }});
                f   ({{ screen.forchheimer[0] }} {{ screen.forchheimer[1] }} {{ screen.forchheimer[2] }});
                
                coordinateSystem
                {
                    type    cartesian;
                    origin  (0 0 0);
                    coordinateRotation
                    {
                        type    axesRotation;
                        e1      (1 0 0);
                        e2      (0 1 0);
                    }
                }
            }
        }
    }
{% endif %}
}

// ************************************************************************* //
"""


class FvOptionsWriter:
    """Generate fvOptions dictionary for momentum sources."""

    def __init__(self, config: Config):
        """
        Initialize fvOptions writer.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.template = Template(FVOPTIONS_TEMPLATE)

    def generate_nozzle_sources(self) -> List[Dict]:
        """Generate nozzle momentum source definitions."""
        if not self.config.geometry.nozzle or not self.config.operation:
            return []
        
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
        
        # Get nozzle zones
        zones = nozzle_array.get_momentum_zones(
            radius=self.config.geometry.nozzle.zone_radius_m,
            height=self.config.geometry.nozzle.zone_height_m,
        )
        
        # Calculate flow per nozzle (m³/s)
        flow_per_nozzle = self.config.operation.pump_total_m3ph / 3600.0 / self.config.geometry.nozzle.count
        
        # Get nozzle diameter in meters
        nozzle_diameter = self.config.geometry.nozzle.throat_diameter_mm / 1000.0
        
        # Build source definitions
        sources = []
        for i, zone in enumerate(zones):
            sources.append({
                "id": f"{i:02d}",
                "name": zone["name"],
                "center": zone["center"],
                "direction": zone["direction"],
                "flow_rate": flow_per_nozzle,
                "diameter": nozzle_diameter,
            })
        
        return sources

    def generate_screen_source(self) -> Optional[Dict]:
        """Generate screen porous zone definition."""
        if not self.config.geometry.screen:
            return None
        
        return {
            "center": self.config.geometry.screen.center,
            "radius": self.config.geometry.screen.radius,
            "height": self.config.geometry.screen.height,
            "darcy": self.config.geometry.screen.darcy,
            "forchheimer": self.config.geometry.screen.forchheimer,
        }

    def write(self, case_dir: Path) -> None:
        """Write fvOptions file to case directory."""
        nozzles = self.generate_nozzle_sources()
        screen = self.generate_screen_source()
        
        content = self.template.render(
            nozzles=nozzles,
            screen=screen,
            rho=self.config.fluid.rho,
        )
        
        output_path = case_dir / "constant" / "fvOptions"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(content)