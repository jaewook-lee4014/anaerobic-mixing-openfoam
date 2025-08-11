"""Calculate pump power and energy consumption."""

from typing import Dict, List, Optional

import numpy as np

from amx.config import Config


class EnergyCalculator:
    """Calculate energy consumption for mixing system."""

    def __init__(self, config: Config):
        """
        Initialize energy calculator.
        
        Args:
            config: Simulation configuration
        """
        self.config = config
        self.g = 9.81  # Gravitational acceleration (m/s²)

    def calculate_pump_power(self, 
                            flow_rate_m3h: Optional[float] = None,
                            head_m: Optional[float] = None,
                            efficiency: Optional[float] = None,
                            include_system_losses: bool = True) -> float:
        """
        Calculate realistic pump power requirement.
        
        P = ρ * g * Q * H_total / η
        
        Where H_total includes:
        - Static head (height difference)
        - Friction losses in pipes
        - Nozzle pressure drop
        - Valve and fitting losses
        
        Args:
            flow_rate_m3h: Flow rate (m³/h)
            head_m: Base head (m) - will add system losses
            efficiency: Pump efficiency (0-1)
            include_system_losses: Include realistic piping losses
            
        Returns:
            Power in watts
        """
        # Use provided values or get from config
        if flow_rate_m3h is None:
            if self.config.operation:
                flow_rate_m3h = self.config.operation.pump_total_m3ph
            else:
                flow_rate_m3h = 430.0  # Default for 32 nozzles
        
        if head_m is None:
            if self.config.operation:
                head_m = self.config.operation.head_m
            else:
                head_m = 10.0  # Default static head
        
        if efficiency is None:
            if self.config.energy:
                efficiency = self.config.energy.pump_efficiency
            else:
                efficiency = 0.65  # Typical for centrifugal pumps
        
        # Calculate system losses for industrial piping
        if include_system_losses:
            # Flow velocity in main pipe (assume 200mm diameter)
            pipe_diameter = 0.2  # m
            pipe_area = np.pi * (pipe_diameter/2)**2
            flow_rate_m3s = flow_rate_m3h / 3600
            pipe_velocity = flow_rate_m3s / pipe_area
            
            # Friction losses (Darcy-Weisbach)
            # Assume 100m total pipe length, friction factor 0.02
            pipe_length = 100  # m
            friction_factor = 0.02
            friction_loss = friction_factor * (pipe_length/pipe_diameter) * (pipe_velocity**2)/(2*self.g)
            
            # Nozzle pressure drop
            # High velocity through small nozzles creates significant loss
            nozzle_velocity = 3.88  # m/s (from jet calculations)
            nozzle_loss = 0.5 * (nozzle_velocity**2) / (2*self.g)  # Loss coefficient ~0.5
            nozzle_loss *= 32  # Total for all nozzles
            
            # Valve and fitting losses (minor losses)
            # Typically 20-30% of friction losses
            minor_losses = 0.25 * friction_loss
            
            # Total dynamic head
            total_head = head_m + friction_loss + nozzle_loss + minor_losses
            
            # Industrial experience: minimum 15-20m for this flow rate
            total_head = max(total_head, 15.0)
        else:
            total_head = head_m
        
        # Convert flow rate to m³/s
        flow_rate_m3s = flow_rate_m3h / 3600
        
        # Fluid properties (sludge, not water)
        rho = self.config.fluid.rho if self.config.fluid else 1015.0  # kg/m³
        
        # Calculate hydraulic power
        hydraulic_power = rho * self.g * flow_rate_m3s * total_head
        
        # Calculate shaft power
        shaft_power = hydraulic_power / max(efficiency, 0.01)
        
        return shaft_power

    def calculate_motor_power(self, shaft_power: Optional[float] = None) -> float:
        """
        Calculate motor power requirement.
        
        Args:
            shaft_power: Pump shaft power (W)
            
        Returns:
            Motor power in watts
        """
        if shaft_power is None:
            shaft_power = self.calculate_pump_power()
        
        # Motor efficiency
        motor_efficiency = 0.90  # Default
        if self.config.energy:
            motor_efficiency = self.config.energy.motor_efficiency
        
        motor_power = shaft_power / max(motor_efficiency, 0.01)
        
        return motor_power

    def calculate_energy_consumption(self,
                                    hours_per_run: float = 2.0,
                                    runs_per_day: int = 3) -> Dict[str, float]:
        """
        Calculate daily energy consumption.
        
        Args:
            hours_per_run: Operating hours per run
            runs_per_day: Number of runs per day
            
        Returns:
            Dictionary with energy metrics
        """
        # Get operation schedule from config if available
        if self.config.operation and self.config.operation.schedule:
            schedule = self.config.operation.schedule
            hours_per_run = schedule.get("hours_per_run", hours_per_run)
            runs_per_day = schedule.get("runs_per_day", runs_per_day)
        
        # Calculate power
        motor_power = self.calculate_motor_power()
        
        # Daily operation
        daily_hours = hours_per_run * runs_per_day
        daily_energy_kwh = motor_power * daily_hours / 1000
        
        # Monthly and annual
        monthly_energy_kwh = daily_energy_kwh * 30
        annual_energy_kwh = daily_energy_kwh * 365
        
        # Energy density
        tank_volume = self.config.geometry.tank.volume
        energy_density_w_m3 = motor_power / tank_volume
        
        return {
            "motor_power_w": motor_power,
            "motor_power_kw": motor_power / 1000,
            "daily_energy_kwh": daily_energy_kwh,
            "monthly_energy_kwh": monthly_energy_kwh,
            "annual_energy_kwh": annual_energy_kwh,
            "energy_density_w_m3": energy_density_w_m3,
            "daily_hours": daily_hours,
            "duty_cycle": daily_hours / 24,
        }

    def calculate_g_value(self, power_w: Optional[float] = None, 
                         use_dissipated_power: bool = True) -> float:
        """
        Calculate velocity gradient (G-value).
        
        G = sqrt(P_dissipated / (μ * V))
        
        Args:
            power_w: Power input (W)
            use_dissipated_power: Use actual dissipated power (not pump power)
            
        Returns:
            G-value in s⁻¹
        """
        if power_w is None:
            # Calculate realistic pump power with losses
            power_w = self.calculate_pump_power(include_system_losses=True)
        
        # For G-value, use power actually dissipated in tank
        # Not all pump power is dissipated as mixing energy
        if use_dissipated_power:
            # Typical efficiency: 60-70% of pump power becomes mixing energy
            # Rest is lost in pipes, nozzles, etc.
            mixing_efficiency = 0.65
            dissipated_power = power_w * mixing_efficiency
        else:
            dissipated_power = power_w
        
        mu = self.config.fluid.mu if self.config.fluid else 0.0035  # Pa·s
        volume = self.config.geometry.tank.volume if self.config.geometry else 2560  # m³
        
        g_value = np.sqrt(dissipated_power / (mu * volume))
        
        return g_value

    def calculate_mixing_reynolds(self) -> float:
        """
        Calculate mixing Reynolds number.
        
        Re = ρ * N * D² / μ
        
        Returns:
            Reynolds number
        """
        # Use jet velocity and nozzle diameter as characteristic values
        if not self.config.operation or not self.config.geometry.nozzle:
            return 0.0
        
        # Characteristic velocity (jet velocity)
        if self.config.operation.jet_velocity_mps:
            U = self.config.operation.jet_velocity_mps
        else:
            # Estimate from flow rate
            Q_nozzle = self.config.operation.pump_total_m3ph / 3600 / self.config.geometry.nozzle.count
            A_nozzle = self.config.geometry.nozzle.throat_area_m2
            U = Q_nozzle / A_nozzle
        
        # Characteristic length (nozzle diameter)
        D = self.config.geometry.nozzle.throat_diameter_mm / 1000
        
        # Calculate Reynolds number
        rho = self.config.fluid.rho
        mu = self.config.fluid.mu
        
        Re = rho * U * D / mu
        
        return Re

    def calculate_power_number(self) -> float:
        """
        Calculate power number for mixing.
        
        Po = P / (ρ * N³ * D⁵)
        
        Returns:
            Power number
        """
        Re = self.calculate_mixing_reynolds()
        
        # Empirical correlation for turbulent regime
        if Re > 10000:
            Po = 5.0  # Typical for turbulent mixing
        elif Re > 100:
            Po = 50 / Re**0.5  # Transitional
        else:
            Po = 70 / Re  # Laminar
        
        return Po

    def compare_scenarios(self, scenarios: List[Dict]) -> Dict[str, Dict]:
        """
        Compare energy consumption for different scenarios.
        
        Args:
            scenarios: List of scenario configurations
            
        Returns:
            Dictionary of scenario comparisons
        """
        results = {}
        
        for scenario in scenarios:
            name = scenario.get("name", "unnamed")
            
            # Calculate energy for scenario
            flow_rate = scenario.get("flow_rate_m3h", self.config.operation.pump_total_m3ph)
            head = scenario.get("head_m", self.config.operation.head_m)
            hours = scenario.get("hours_per_run", 2.0)
            runs = scenario.get("runs_per_day", 3)
            
            power = self.calculate_pump_power(flow_rate, head)
            motor_power = power / 0.9  # Assume 90% motor efficiency
            
            daily_kwh = motor_power * hours * runs / 1000
            annual_kwh = daily_kwh * 365
            
            # G-value for scenario
            g_value = np.sqrt(power / (self.config.fluid.mu * self.config.geometry.tank.volume))
            
            results[name] = {
                "flow_rate_m3h": flow_rate,
                "head_m": head,
                "pump_power_w": power,
                "motor_power_w": motor_power,
                "daily_energy_kwh": daily_kwh,
                "annual_energy_kwh": annual_kwh,
                "g_value": g_value,
                "operating_hours_per_day": hours * runs,
            }
        
        # Add comparison metrics
        if results:
            baseline = list(results.values())[0]
            for name, data in results.items():
                data["energy_ratio"] = data["annual_energy_kwh"] / baseline["annual_energy_kwh"]
                data["power_ratio"] = data["motor_power_w"] / baseline["motor_power_w"]
        
        return results

    def optimize_operation(self, 
                          target_g_value: float = 50,
                          max_power_w: float = 50000) -> Dict:
        """
        Optimize operation parameters for target mixing.
        
        Args:
            target_g_value: Target velocity gradient (s⁻¹)
            max_power_w: Maximum allowable power (W)
            
        Returns:
            Optimized operation parameters
        """
        mu = self.config.fluid.mu
        volume = self.config.geometry.tank.volume
        
        # Required power for target G-value
        required_power = (target_g_value**2) * mu * volume
        
        if required_power > max_power_w:
            # Reduce G-value to stay within power limit
            achievable_g = np.sqrt(max_power_w / (mu * volume))
            actual_power = max_power_w
        else:
            achievable_g = target_g_value
            actual_power = required_power
        
        # Back-calculate flow rate and head
        # P = ρ * g * Q * H / η
        rho = self.config.fluid.rho
        eta = 0.65
        
        # Assume head is fixed, calculate required flow
        if self.config.operation:
            head = self.config.operation.head_m
        else:
            head = 15.0  # Default
        
        required_flow_m3s = actual_power * eta / (rho * self.g * head)
        required_flow_m3h = required_flow_m3s * 3600
        
        # Calculate operating schedule for energy efficiency
        # Balance between continuous low-power and intermittent high-power
        if achievable_g < 30:
            # Low mixing - intermittent operation
            hours_per_run = 2
            runs_per_day = 3
        elif achievable_g < 50:
            # Medium mixing
            hours_per_run = 3
            runs_per_day = 4
        else:
            # High mixing - nearly continuous
            hours_per_run = 6
            runs_per_day = 3
        
        return {
            "target_g_value": target_g_value,
            "achievable_g_value": achievable_g,
            "required_power_w": actual_power,
            "required_flow_m3h": required_flow_m3h,
            "head_m": head,
            "hours_per_run": hours_per_run,
            "runs_per_day": runs_per_day,
            "daily_energy_kwh": actual_power * hours_per_run * runs_per_day / 1000,
            "optimization_feasible": required_power <= max_power_w,
        }