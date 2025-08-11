"""Geometry parameterization for tank and nozzles."""

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass
class Point3D:
    """3D point representation."""

    x: float
    y: float
    z: float

    def __repr__(self) -> str:
        return f"({self.x:.3f}, {self.y:.3f}, {self.z:.3f})"

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def to_list(self) -> List[float]:
        return [self.x, self.y, self.z]


@dataclass
class Vector3D:
    """3D vector representation."""

    x: float
    y: float
    z: float

    def normalize(self) -> "Vector3D":
        """Return normalized vector."""
        mag = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if mag > 0:
            return Vector3D(self.x / mag, self.y / mag, self.z / mag)
        return Vector3D(0, 0, 0)

    def magnitude(self) -> float:
        """Return vector magnitude."""
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.x, self.y, self.z)


class TankGeometry:
    """Tank geometry definition."""

    def __init__(self, L: float, W: float, H: float):
        """
        Initialize tank geometry.
        
        Args:
            L: Tank length (m)
            W: Tank width (m)
            H: Tank height (m)
        """
        self.L = L
        self.W = W
        self.H = H

    @property
    def volume(self) -> float:
        """Calculate tank volume in m³."""
        return self.L * self.W * self.H

    @property
    def surface_area(self) -> float:
        """Calculate tank surface area in m²."""
        return 2 * (self.L * self.W + self.L * self.H + self.W * self.H)

    @property
    def hydraulic_diameter(self) -> float:
        """Calculate hydraulic diameter."""
        perimeter = 2 * (self.L + self.W)
        area = self.L * self.W
        return 4 * area / perimeter

    def get_corners(self) -> List[Point3D]:
        """Get tank corner points."""
        return [
            Point3D(0, 0, 0),
            Point3D(self.L, 0, 0),
            Point3D(self.L, self.W, 0),
            Point3D(0, self.W, 0),
            Point3D(0, 0, self.H),
            Point3D(self.L, 0, self.H),
            Point3D(self.L, self.W, self.H),
            Point3D(0, self.W, self.H),
        ]

    def get_center(self) -> Point3D:
        """Get tank center point."""
        return Point3D(self.L / 2, self.W / 2, self.H / 2)


class NozzleArray:
    """Nozzle array configuration and positioning."""

    def __init__(
        self,
        count: int,
        array: Tuple[int, int],
        tank: TankGeometry,
        pitch_deg: float = 45.0,
        start_offset: Tuple[float, float] = (1.0, 1.0),
        spacing: Tuple[float, float] = None,
        z_position: float = 0.5,
    ):
        """
        Initialize nozzle array.
        
        Args:
            count: Number of nozzles
            array: Array configuration [rows, cols]
            tank: Tank geometry
            pitch_deg: Upward pitch angle (degrees)
            start_offset: Offset from walls [x, y] (m)
            spacing: Spacing between nozzles [x, y] (m)
            z_position: Nozzle height from bottom (m)
        """
        self.count = count
        self.rows, self.cols = array
        self.tank = tank
        self.pitch_deg = pitch_deg
        self.pitch_rad = math.radians(pitch_deg)
        self.start_offset = start_offset
        self.z_position = z_position

        # Calculate spacing if not provided
        if spacing is None:
            self.spacing = self._calculate_spacing()
        else:
            self.spacing = spacing

        # Validate configuration
        if self.rows * self.cols != self.count:
            raise ValueError(f"Array {array} doesn't match count {count}")

    def _calculate_spacing(self) -> Tuple[float, float]:
        """Calculate uniform spacing between nozzles."""
        available_x = self.tank.L - 2 * self.start_offset[0]
        available_y = self.tank.W - 2 * self.start_offset[1]

        spacing_x = available_x / (self.rows - 1) if self.rows > 1 else 0
        spacing_y = available_y / (self.cols - 1) if self.cols > 1 else 0

        return (spacing_x, spacing_y)

    def get_positions(self) -> List[Point3D]:
        """Get nozzle positions."""
        positions = []
        for i in range(self.rows):
            for j in range(self.cols):
                x = self.start_offset[0] + i * self.spacing[0]
                y = self.start_offset[1] + j * self.spacing[1]
                positions.append(Point3D(x, y, self.z_position))
        return positions

    def get_directions(self) -> List[Vector3D]:
        """Get nozzle jet directions (45° upward toward center)."""
        directions = []
        tank_center = self.tank.get_center()

        for pos in self.get_positions():
            # Vector toward tank center
            to_center_x = tank_center.x - pos.x
            to_center_y = tank_center.y - pos.y

            # Normalize horizontal component
            horiz_mag = math.sqrt(to_center_x**2 + to_center_y**2)
            if horiz_mag > 0:
                to_center_x /= horiz_mag
                to_center_y /= horiz_mag

            # Apply pitch angle (45° upward)
            dir_x = to_center_x * math.cos(self.pitch_rad)
            dir_y = to_center_y * math.cos(self.pitch_rad)
            dir_z = math.sin(self.pitch_rad)

            directions.append(Vector3D(dir_x, dir_y, dir_z).normalize())

        return directions

    def get_momentum_zones(
        self, radius: float = 0.2, height: float = 0.5
    ) -> List[dict]:
        """
        Get momentum source zone definitions.
        
        Args:
            radius: Zone radius (m)
            height: Zone height (m)
            
        Returns:
            List of zone dictionaries with position and dimensions
        """
        zones = []
        positions = self.get_positions()
        directions = self.get_directions()

        for i, (pos, dir_vec) in enumerate(zip(positions, directions)):
            # Zone extends along jet direction
            zone_center = Point3D(
                pos.x + dir_vec.x * height / 2,
                pos.y + dir_vec.y * height / 2,
                pos.z + dir_vec.z * height / 2,
            )

            zones.append(
                {
                    "name": f"nozzle_{i:02d}",
                    "center": zone_center.to_tuple(),
                    "radius": radius,
                    "height": height,
                    "direction": dir_vec.to_tuple(),
                    "position": pos.to_tuple(),
                }
            )

        return zones


class ScreenGeometry:
    """Screen/porous zone geometry."""

    def __init__(
        self,
        center: Tuple[float, float, float],
        radius: float,
        height: float,
    ):
        """
        Initialize screen geometry.
        
        Args:
            center: Screen center position [x, y, z] (m)
            radius: Screen radius (m)
            height: Screen height (m)
        """
        self.center = Point3D(*center)
        self.radius = radius
        self.height = height

    @property
    def volume(self) -> float:
        """Calculate screen volume in m³."""
        return math.pi * self.radius**2 * self.height

    @property
    def surface_area(self) -> float:
        """Calculate screen surface area in m²."""
        return 2 * math.pi * self.radius * self.height

    def get_bounding_box(self) -> Tuple[Point3D, Point3D]:
        """Get bounding box corners."""
        min_point = Point3D(
            self.center.x - self.radius,
            self.center.y - self.radius,
            self.center.z - self.height / 2,
        )
        max_point = Point3D(
            self.center.x + self.radius,
            self.center.y + self.radius,
            self.center.z + self.height / 2,
        )
        return (min_point, max_point)