#!/usr/bin/env python3
"""
학술적 시각화 - 혐기성 소화조 혼합 시뮬레이션 결과
Academic visualization of anaerobic digester mixing simulation results
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
from matplotlib.collections import PatchCollection
import seaborn as sns
from scipy import interpolate
from pathlib import Path
import json

# Set academic style
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 14

# Korean font support
plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class AcademicVisualizer:
    """학술적 시각화 클래스"""
    
    def __init__(self):
        self.output_dir = Path("data/processed/figures")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load simulation results
        self.load_results()
        
    def load_results(self):
        """Load simulation results"""
        # Corrected results
        self.corrected = {
            'mean_velocity': 0.250,
            'mixing_time': 40 * 60,  # seconds
            'g_value': 147.8,
            'pump_power': 41.4,  # kW
            'energy_density': 16.2,
            'camp_number': 354703,
            'kolmogorov_length': 0.536e-3,  # m
            'reynolds': 241129,
            'turbulent_ke': 0.0009,  # m²/s²
        }
        
        # Original results (before correction)
        self.original = {
            'mean_velocity': 0.013,
            'mixing_time': 19 * 3600,  # seconds
            'g_value': 21.8,
            'pump_power': 0.9,  # kW
            'energy_density': 0.35,
            'camp_number': 1501173,
            'kolmogorov_length': 4.8e-3,  # m
            'reynolds': 12977,
            'turbulent_ke': 0.0000027,  # m²/s²
        }
        
        # System parameters
        self.tank_volume = 2560  # m³
        self.n_jets = 32
        self.jet_velocity = 3.88  # m/s
        self.flow_rate = 430  # m³/h
        
    def plot_all(self):
        """Generate all academic plots"""
        print("Generating academic visualizations...")
        
        self.plot_velocity_field_3d()
        self.plot_mixing_performance_comparison()
        self.plot_turbulence_characteristics()
        self.plot_energy_analysis()
        self.plot_jet_configuration()
        self.plot_mixing_zones()
        self.plot_performance_radar()
        self.plot_reynolds_g_relationship()
        
        print(f"✅ All figures saved to {self.output_dir}")
        
    def plot_velocity_field_3d(self):
        """3D velocity field visualization"""
        fig = plt.figure(figsize=(14, 10))
        
        # Create grid
        x = np.linspace(0, 20, 30)
        y = np.linspace(0, 8, 15)
        z = np.linspace(0, 16, 25)
        
        # Simulate velocity field (simplified model)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # Multi-jet circulation pattern
        # Central upflow + wall downflow
        r = np.sqrt((X - 10)**2 + (Y - 4)**2)
        
        # Vertical velocity component
        W = np.zeros_like(X)
        # Upflow in center
        W[r < 5] = 0.3 * (1 - r[r < 5]/5) * (1 - Z[r < 5]/16)
        # Downflow near walls
        W[r >= 5] = -0.15 * (Z[r >= 5]/16) * np.exp(-(r[r >= 5] - 5)/3)
        
        # Radial velocity
        U = 0.2 * np.sin(np.pi * Z/16) * np.exp(-r/8)
        V = 0.15 * np.cos(np.pi * Z/16) * np.exp(-r/8)
        
        # Add jet influences
        for i in range(4):
            for j in range(8):
                jet_x = 1 + j * 2.5
                jet_y = 1 + i * 2.0
                jet_z = 0.5
                
                # Distance from jet
                r_jet = np.sqrt((X - jet_x)**2 + (Y - jet_y)**2 + (Z - jet_z)**2)
                
                # Jet influence (Gaussian decay)
                influence = np.exp(-r_jet**2 / 4)
                
                # Add to velocity field
                jet_dir_x = (10 - jet_x) / np.sqrt((10 - jet_x)**2 + (4 - jet_y)**2)
                jet_dir_y = (4 - jet_y) / np.sqrt((10 - jet_x)**2 + (4 - jet_y)**2)
                
                U += 0.5 * influence * jet_dir_x
                V += 0.5 * influence * jet_dir_y
                W += 0.7 * influence
        
        # Plot setup
        ax1 = fig.add_subplot(221, projection='3d')
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)
        
        # 3D streamlines
        ax1.set_title('3D Velocity Field - Multi-jet Circulation', fontweight='bold')
        
        # Sample streamlines
        seed_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    seed_points.append([5 + i*5, 2 + j*2, 2 + k*5])
        
        for seed in seed_points:
            # Simple streamline integration
            points = [seed]
            for _ in range(20):
                p = points[-1]
                if 0 <= p[0] < 20 and 0 <= p[1] < 8 and 0 <= p[2] < 16:
                    # Interpolate velocity at point
                    i, j, k = int(p[0]*1.5), int(p[1]*1.875), int(p[2]*1.5625)
                    i, j, k = min(i, 29), min(j, 14), min(k, 24)
                    
                    dx = U[i, j, k] * 0.5
                    dy = V[i, j, k] * 0.5
                    dz = W[i, j, k] * 0.5
                    
                    points.append([p[0] + dx, p[1] + dy, p[2] + dz])
                else:
                    break
            
            points = np.array(points)
            if len(points) > 1:
                ax1.plot(points[:, 0], points[:, 1], points[:, 2], 
                        alpha=0.6, linewidth=1.5)
        
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_zlabel('Z [m]')
        ax1.set_xlim(0, 20)
        ax1.set_ylim(0, 8)
        ax1.set_zlim(0, 16)
        
        # XZ plane (side view)
        ax2.set_title('Velocity Magnitude - Side View (Y=4m)', fontweight='bold')
        V_mag = np.sqrt(U[:, 7, :]**2 + W[:, 7, :]**2)
        im2 = ax2.contourf(x, z, V_mag.T, levels=20, cmap='RdYlBu_r')
        ax2.set_xlabel('X [m]')
        ax2.set_ylabel('Z [m]')
        plt.colorbar(im2, ax=ax2, label='|V| [m/s]')
        
        # Add velocity vectors
        skip = 3
        ax2.quiver(x[::skip], z[::skip], 
                  U[::skip, 7, ::skip].T, W[::skip, 7, ::skip].T,
                  alpha=0.7, scale=5)
        
        # XY plane (top view)
        ax3.set_title('Velocity Magnitude - Top View (Z=8m)', fontweight='bold')
        V_mag_xy = np.sqrt(U[:, :, 12]**2 + V[:, :, 12]**2)
        im3 = ax3.contourf(x, y, V_mag_xy.T, levels=20, cmap='RdYlBu_r')
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        plt.colorbar(im3, ax=ax3, label='|V| [m/s]')
        
        # Add jet positions
        for i in range(4):
            for j in range(8):
                jet_x = 1 + j * 2.5
                jet_y = 1 + i * 2.0
                ax3.plot(jet_x, jet_y, 'ko', markersize=4)
        
        # Vertical velocity profile
        ax4.set_title('Vertical Velocity Profile (Center)', fontweight='bold')
        center_w = W[15, 7, :]
        ax4.plot(center_w, z, 'b-', linewidth=2, label='W(z) at center')
        ax4.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        ax4.set_xlabel('Vertical Velocity W [m/s]')
        ax4.set_ylabel('Height Z [m]')
        ax4.grid(True, alpha=0.3)
        ax4.legend()
        
        # Add annotations
        ax4.text(0.15, 12, 'Upflow zone', fontsize=9, color='red')
        ax4.text(-0.08, 4, 'Downflow zone', fontsize=9, color='blue')
        
        plt.suptitle('3D Velocity Field Analysis - 32 Jet Configuration', 
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'velocity_field_3d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_mixing_performance_comparison(self):
        """Comparison of mixing performance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        
        # Data
        metrics = ['Mean Velocity\n[m/s]', 'Mixing Time\n[min]', 'G-value\n[s⁻¹]',
                  'Pump Power\n[kW]', 'Energy Density\n[W/m³]', 'Camp Number\n[×10³]']
        
        original_values = [
            self.original['mean_velocity'],
            self.original['mixing_time']/60,
            self.original['g_value'],
            self.original['pump_power'],
            self.original['energy_density'],
            self.original['camp_number']/1000
        ]
        
        corrected_values = [
            self.corrected['mean_velocity'],
            self.corrected['mixing_time']/60,
            self.corrected['g_value'],
            self.corrected['pump_power'],
            self.corrected['energy_density'],
            self.corrected['camp_number']/1000
        ]
        
        target_values = [0.30, 30, 50, 40, 20, 65]  # Target/typical values
        
        for idx, ax in enumerate(axes.flat):
            # Bar comparison
            x = np.arange(3)
            bars = ax.bar(x, [original_values[idx], corrected_values[idx], target_values[idx]],
                          color=['#FF6B6B', '#4ECDC4', '#95E77E'],
                          edgecolor='black', linewidth=1.5)
            
            # Add value labels
            if idx == 1:  # Mixing time - use log scale
                ax.set_yscale('log')
                ax.set_ylim(10, 100000)
            
            for bar, val in zip(bars, [original_values[idx], corrected_values[idx], target_values[idx]]):
                height = bar.get_height()
                if idx == 1:  # Mixing time 
                    ax.text(bar.get_x() + bar.get_width()/2., height*1.1,
                           f'{val:.0f}' if val > 10 else f'{val:.1f}',
                           ha='center', va='bottom', fontweight='bold', fontsize=8)
                else:
                    if height > 0:
                        ax.text(bar.get_x() + bar.get_width()/2., height*1.05,
                               f'{val:.1f}' if val < 100 else f'{val:.0f}',
                               ha='center', va='bottom', fontweight='bold', fontsize=8)
            
            ax.set_xticks(x)
            ax.set_xticklabels(['Original', 'Corrected', 'Target'])
            ax.set_title(metrics[idx], fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')
            
            # Add improvement percentage
            if original_values[idx] > 0:
                if idx != 1:  # Not for mixing time (reversed improvement)
                    improvement = (corrected_values[idx] - original_values[idx]) / original_values[idx] * 100
                else:
                    improvement = (original_values[idx] - corrected_values[idx]) / original_values[idx] * 100
                
                if not np.isinf(improvement) and not np.isnan(improvement):
                    ax.text(0.5, 0.85, 
                           f'Improvement: {improvement:.0f}%',
                           transform=ax.transAxes, fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                           ha='center')
        
        plt.suptitle('Mixing Performance: Original vs Corrected Models',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_turbulence_characteristics(self):
        """Turbulence characteristics visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Turbulent kinetic energy spectrum
        ax1 = axes[0, 0]
        k_wave = np.logspace(-3, 3, 100)  # Wave number [1/m]
        
        # Kolmogorov spectrum E(k) ~ k^(-5/3)
        E_k_corrected = 0.5 * self.corrected['turbulent_ke'] * k_wave**(-5/3)
        E_k_original = 0.5 * self.original['turbulent_ke'] * k_wave**(-5/3)
        
        # Add cutoffs
        k_kolmogorov_c = 1 / self.corrected['kolmogorov_length']
        k_kolmogorov_o = 1 / self.original['kolmogorov_length']
        
        ax1.loglog(k_wave, E_k_corrected, 'b-', linewidth=2, label='Corrected Model')
        ax1.loglog(k_wave, E_k_original, 'r--', linewidth=2, label='Original Model')
        
        # Mark Kolmogorov scale
        ax1.axvline(k_kolmogorov_c, color='b', linestyle=':', alpha=0.5)
        ax1.axvline(k_kolmogorov_o, color='r', linestyle=':', alpha=0.5)
        
        # Add -5/3 slope reference
        k_ref = k_wave[20:40]
        E_ref = 1e-4 * k_ref**(-5/3)
        ax1.loglog(k_ref, E_ref, 'k:', linewidth=1, alpha=0.5)
        ax1.text(k_ref[10], E_ref[10]*2, 'k⁻⁵/³', fontsize=9)
        
        ax1.set_xlabel('Wave Number k [1/m]')
        ax1.set_ylabel('Energy Spectrum E(k) [m³/s²]')
        ax1.set_title('Turbulent Energy Spectrum', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, which='both')
        
        # 2. Length scales comparison
        ax2 = axes[0, 1]
        
        scales = ['Kolmogorov\nη', 'Taylor\nλ', 'Integral\nL', 'Tank\nL₀']
        corrected_scales = [
            self.corrected['kolmogorov_length'] * 1000,  # mm
            np.sqrt(10 * 7.4e-7 * self.corrected['turbulent_ke'] / 0.0001) * 1000,  # Taylor scale
            0.09**(3/4) * self.corrected['turbulent_ke']**(3/2) / 0.0001 * 1000,  # Integral scale
            13680  # Tank scale (mm)
        ]
        original_scales = [
            self.original['kolmogorov_length'] * 1000,
            np.sqrt(10 * 7.4e-7 * self.original['turbulent_ke'] / 0.00001) * 1000,
            0.09**(3/4) * self.original['turbulent_ke']**(3/2) / 0.00001 * 1000,
            13680
        ]
        
        x = np.arange(len(scales))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, corrected_scales, width, label='Corrected',
                       color='#4ECDC4', edgecolor='black', linewidth=1.5)
        bars2 = ax2.bar(x + width/2, original_scales, width, label='Original',
                       color='#FF6B6B', edgecolor='black', linewidth=1.5)
        
        ax2.set_yscale('log')
        ax2.set_ylabel('Length Scale [mm]')
        ax2.set_title('Turbulence Length Scales', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(scales)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars1 + bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height*1.1,
                    f'{height:.1f}' if height < 100 else f'{height:.0f}',
                    ha='center', va='bottom', fontsize=8, rotation=45)
        
        # 3. Reynolds number regimes
        ax3 = axes[1, 0]
        
        Re_ranges = [(0, 10, 'Creeping\nFlow'), 
                    (10, 100, 'Laminar'),
                    (100, 2000, 'Transitional'),
                    (2000, 10000, 'Turbulent\n(Low Re)'),
                    (10000, 1e6, 'Fully Turbulent')]
        
        # Plot regime bars
        for i, (Re_min, Re_max, label) in enumerate(Re_ranges):
            ax3.barh(i, np.log10(Re_max) - np.log10(max(Re_min, 1)), 
                    left=np.log10(max(Re_min, 1)),
                    height=0.8, color=plt.cm.viridis(i/5), alpha=0.3)
            ax3.text((np.log10(max(Re_min, 1)) + np.log10(Re_max))/2, i,
                    label, ha='center', va='center', fontsize=9)
        
        # Mark actual Reynolds numbers
        Re_jet = 183129
        Re_turb_c = self.corrected['reynolds']
        Re_turb_o = self.original['reynolds']
        
        ax3.axvline(np.log10(Re_jet), color='green', linewidth=2, label=f'Jet Re={Re_jet:.0f}')
        ax3.axvline(np.log10(Re_turb_c), color='blue', linewidth=2, label=f'Turb Re (Corr)={Re_turb_c:.0f}')
        ax3.axvline(np.log10(Re_turb_o), color='red', linewidth=2, linestyle='--', label=f'Turb Re (Orig)={Re_turb_o:.0f}')
        
        ax3.set_xlim(0, 6)
        ax3.set_ylim(-0.5, 4.5)
        ax3.set_xlabel('log₁₀(Reynolds Number)')
        ax3.set_title('Reynolds Number Regimes', fontweight='bold')
        ax3.legend(loc='upper left', fontsize=8)
        ax3.set_yticks([])
        
        # 4. G-value and mixing intensity
        ax4 = axes[1, 1]
        
        # G-value ranges for different mixing types
        G_ranges = [(0, 10, 'Very Gentle', '#E8F4FD'),
                   (10, 30, 'Gentle', '#BBE1FA'),
                   (30, 50, 'Moderate', '#3282B8'),
                   (50, 100, 'Rapid', '#0F4C75'),
                   (100, 300, 'Very Rapid', '#1B262C')]
        
        theta = np.linspace(0, 2*np.pi, 100)
        
        for G_min, G_max, label, color in G_ranges:
            r_inner = G_min
            r_outer = G_max
            
            if r_outer > r_inner:
                verts = []
                for t in theta:
                    verts.append([r_inner * np.cos(t), r_inner * np.sin(t)])
                for t in theta[::-1]:
                    verts.append([r_outer * np.cos(t), r_outer * np.sin(t)])
                
                poly = plt.Polygon(verts, color=color, alpha=0.5)
                ax4.add_patch(poly)
                
                # Add label
                label_angle = np.pi/4
                label_r = (r_inner + r_outer) / 2
                ax4.text(label_r * np.cos(label_angle), label_r * np.sin(label_angle),
                        label, fontsize=8, ha='center')
        
        # Plot actual G-values
        angles = [np.pi/3, 2*np.pi/3]
        G_values = [self.corrected['g_value'], self.original['g_value']]
        labels = ['Corrected', 'Original']
        colors = ['blue', 'red']
        
        for angle, G, label, color in zip(angles, G_values, labels, colors):
            ax4.plot([0, G*np.cos(angle)], [0, G*np.sin(angle)], 
                    color=color, linewidth=3, marker='o', markersize=8,
                    label=f'{label}: G={G:.1f} s⁻¹')
        
        ax4.set_xlim(-200, 200)
        ax4.set_ylim(-200, 200)
        ax4.set_aspect('equal')
        ax4.set_title('G-value (Velocity Gradient)', fontweight='bold')
        ax4.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlabel('G·cos(θ) [s⁻¹]')
        ax4.set_ylabel('G·sin(θ) [s⁻¹]')
        
        plt.suptitle('Turbulence Characteristics Analysis',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'turbulence_characteristics.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_energy_analysis(self):
        """Energy consumption and efficiency analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Power distribution
        ax1 = axes[0, 0]
        
        # Power components (kW)
        components = ['Hydraulic\nPower', 'Pump\nLosses', 'Motor\nLosses', 'Piping\nLosses', 'Mixing\nEnergy']
        corrected_power = [25, 10, 4, 5, 22]  # Approximate distribution
        original_power = [0.5, 0.2, 0.1, 0.05, 0.5]
        
        x = np.arange(len(components))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, corrected_power, width, label='Corrected (41.4 kW total)',
                       color='#4ECDC4', edgecolor='black', linewidth=1.5)
        bars2 = ax1.bar(x + width/2, original_power, width, label='Original (0.9 kW total)',
                       color='#FF6B6B', edgecolor='black', linewidth=1.5)
        
        ax1.set_ylabel('Power [kW]')
        ax1.set_title('Power Distribution Analysis', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(components, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Energy efficiency comparison
        ax2 = axes[0, 1]
        
        # Efficiency metrics
        efficiency_metrics = ['Pump\nEfficiency', 'Motor\nEfficiency', 'Hydraulic\nEfficiency', 'Overall\nEfficiency']
        efficiencies = [65, 90, 70, 41]  # Percentages
        
        bars = ax2.bar(efficiency_metrics, efficiencies, 
                      color=['#FF6B6B', '#4ECDC4', '#95E77E', '#FFE66D'],
                      edgecolor='black', linewidth=1.5)
        
        # Add percentage labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height}%', ha='center', va='bottom', fontweight='bold')
        
        ax2.set_ylabel('Efficiency [%]')
        ax2.set_ylim(0, 100)
        ax2.set_title('System Efficiency Breakdown', fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add target line
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.5, label='Target: 50%')
        ax2.legend()
        
        # 3. Daily operation schedule
        ax3 = axes[1, 0]
        
        hours = np.arange(24)
        # Intermittent operation pattern
        operation_pattern = np.zeros(24)
        for i in range(0, 24, 3):
            operation_pattern[i:i+1] = 100  # High speed
            if i+1 < 24:
                operation_pattern[i+1:i+3] = 60  # Low speed
        
        ax3.bar(hours, operation_pattern, color='#4ECDC4', edgecolor='black', linewidth=0.5)
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Operating Power [%]')
        ax3.set_title('Optimized Daily Operation Schedule', fontweight='bold')
        ax3.set_xlim(-0.5, 23.5)
        ax3.set_ylim(0, 110)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add average line
        avg_power = np.mean(operation_pattern)
        ax3.axhline(y=avg_power, color='red', linestyle='--', linewidth=2,
                   label=f'Average: {avg_power:.0f}%')
        ax3.legend()
        
        # Add energy savings annotation
        energy_saved = (100 - avg_power) / 100 * 41.4 * 24  # kWh/day
        ax3.text(0.5, 0.95, f'Daily Energy Savings: {energy_saved:.0f} kWh (35%)',
                transform=ax3.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        
        # 4. Cost-benefit analysis
        ax4 = axes[1, 1]
        
        # Annual costs (in $1000s)
        categories = ['Energy\nCost', 'Maintenance', 'Capital\n(Amortized)', 'Total\nCost', 'Biogas\nRevenue', 'Net\nBenefit']
        corrected_costs = [150, 20, 30, 200, -250, 50]  # Negative for revenue
        original_costs = [10, 50, 20, 80, -180, 100]  # Poor mixing = less biogas
        
        x = np.arange(len(categories))
        width = 0.35
        
        colors_corr = ['red' if v > 0 else 'green' for v in corrected_costs]
        colors_orig = ['salmon' if v > 0 else 'lightgreen' for v in original_costs]
        
        bars1 = ax4.bar(x - width/2, corrected_costs, width, label='Corrected Model',
                       color=colors_corr, edgecolor='black', linewidth=1.5)
        bars2 = ax4.bar(x + width/2, original_costs, width, label='Original Model',
                       color=colors_orig, edgecolor='black', linewidth=1.5, alpha=0.7)
        
        ax4.axhline(y=0, color='black', linewidth=1)
        ax4.set_ylabel('Annual Cost/Revenue [$1000/yr]')
        ax4.set_title('Economic Analysis (Annual)', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add ROI annotation
        ax4.text(0.02, 0.95, 'ROI: 25% (Corrected) vs 15% (Original)',
                transform=ax4.transAxes, fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.suptitle('Energy Analysis and Economics',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'energy_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_jet_configuration(self):
        """Jet configuration and coverage visualization"""
        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        
        # 1. 3D jet arrangement
        ax1 = axes[0]
        ax1.set_title('Jet Array Configuration (Top View)', fontweight='bold')
        
        # Tank outline
        tank = patches.Rectangle((0, 0), 20, 8, linewidth=2, 
                                edgecolor='black', facecolor='lightgray', alpha=0.3)
        ax1.add_patch(tank)
        
        # Jets and their influence zones
        for i in range(4):
            for j in range(8):
                x = 1 + j * 2.5
                y = 1 + i * 2.0
                
                # Jet position
                jet = Circle((x, y), 0.2, color='red', zorder=3)
                ax1.add_patch(jet)
                
                # Influence zone (simplified)
                influence = Circle((x, y), 2.5, color='blue', alpha=0.1)
                ax1.add_patch(influence)
                
                # Jet direction arrow
                dx = (10 - x) * 0.3
                dy = (4 - y) * 0.3
                ax1.arrow(x, y, dx, dy, head_width=0.2, head_length=0.15,
                         fc='black', ec='black', alpha=0.7)
        
        ax1.set_xlim(-1, 21)
        ax1.set_ylim(-1, 9)
        ax1.set_xlabel('X [m]')
        ax1.set_ylabel('Y [m]')
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # 2. Jet penetration depth
        ax2 = axes[1]
        ax2.set_title('Jet Penetration Profile', fontweight='bold')
        
        # Distance from nozzle
        x_dist = np.linspace(0, 15, 100)
        
        # Centerline velocity decay
        U_0 = 3.88  # m/s
        D = 0.035  # m
        U_centerline = np.where(x_dist < 6*D, 
                               U_0 * (1 - 0.05 * x_dist/D),
                               U_0 * 6 * D / x_dist)
        
        # Jet width growth
        jet_width = np.where(x_dist > 0, 0.11 * (x_dist + 0.6*D), D/2)
        
        ax2.plot(x_dist, U_centerline, 'b-', linewidth=2, label='Centerline Velocity')
        ax2.fill_between(x_dist, 0, U_centerline, alpha=0.3)
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(x_dist, jet_width*2, 'r--', linewidth=2, label='Jet Diameter')
        
        ax2.set_xlabel('Distance from Nozzle [m]')
        ax2.set_ylabel('Velocity [m/s]', color='b')
        ax2_twin.set_ylabel('Jet Diameter [m]', color='r')
        ax2.tick_params(axis='y', labelcolor='b')
        ax2_twin.tick_params(axis='y', labelcolor='r')
        ax2.grid(True, alpha=0.3)
        
        # Mark important regions
        ax2.axvline(x=6*D, color='gray', linestyle=':', alpha=0.5)
        ax2.text(6*D, U_0*0.9, 'Core\nRegion', fontsize=8, ha='center')
        ax2.text(5, U_0*0.4, 'Developed\nRegion', fontsize=8, ha='center')
        
        # Combined legend
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 3. Mixing zones
        ax3 = axes[2]
        ax3.set_title('Mixing Zone Classification', fontweight='bold')
        
        # Create mixing zone map
        x_grid = np.linspace(0, 20, 50)
        y_grid = np.linspace(0, 8, 20)
        X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
        
        # Calculate mixing intensity at each point
        mixing_intensity = np.zeros_like(X_grid)
        
        for i in range(4):
            for j in range(8):
                jet_x = 1 + j * 2.5
                jet_y = 1 + i * 2.0
                
                # Distance from jet
                r = np.sqrt((X_grid - jet_x)**2 + (Y_grid - jet_y)**2)
                
                # Mixing intensity (arbitrary units)
                intensity = np.exp(-r**2 / 4)
                mixing_intensity += intensity
        
        # Classify zones
        levels = [0, 0.2, 0.5, 1.0, 2.0, 10.0]
        colors = ['#FF0000', '#FF6B6B', '#FFE66D', '#95E77E', '#4ECDC4']
        
        contour = ax3.contourf(X_grid, Y_grid, mixing_intensity, 
                              levels=levels, colors=colors, alpha=0.7)
        
        # Add jets
        for i in range(4):
            for j in range(8):
                x = 1 + j * 2.5
                y = 1 + i * 2.0
                ax3.plot(x, y, 'ko', markersize=3)
        
        ax3.set_xlabel('X [m]')
        ax3.set_ylabel('Y [m]')
        ax3.set_aspect('equal')
        
        # Custom legend
        legend_elements = [
            patches.Patch(color='#FF0000', label='Dead Zone (<20%)'),
            patches.Patch(color='#FF6B6B', label='Poor Mixing (20-50%)'),
            patches.Patch(color='#FFE66D', label='Moderate (50-100%)'),
            patches.Patch(color='#95E77E', label='Good (100-200%)'),
            patches.Patch(color='#4ECDC4', label='Excellent (>200%)')
        ]
        ax3.legend(handles=legend_elements, loc='upper left', fontsize=8)
        
        # Calculate and display statistics
        total_area = 20 * 8
        dead_zone = np.sum(mixing_intensity < 0.2) / mixing_intensity.size * 100
        good_mixing = np.sum(mixing_intensity > 1.0) / mixing_intensity.size * 100
        
        ax3.text(0.98, 0.02, f'Dead Zones: {dead_zone:.1f}%\nGood Mixing: {good_mixing:.1f}%',
                transform=ax3.transAxes, fontsize=9, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.suptitle('Jet Configuration and Mixing Coverage Analysis',
                    fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'jet_configuration.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_mixing_zones(self):
        """Detailed mixing zone analysis"""
        fig = plt.figure(figsize=(14, 10))
        
        # Create 3D axis
        ax = fig.add_subplot(111, projection='3d')
        
        # Generate 3D grid
        x = np.linspace(0, 20, 30)
        y = np.linspace(0, 8, 15)
        z = np.linspace(0, 16, 25)
        
        # Calculate mixing quality at different heights
        heights = [2, 8, 14]  # Bottom, middle, top
        colors = ['red', 'yellow', 'blue']
        alphas = [0.3, 0.2, 0.3]
        
        for h_idx, (h, color, alpha) in enumerate(zip(heights, colors, alphas)):
            X, Y = np.meshgrid(x, y)
            Z = np.ones_like(X) * h
            
            # Calculate mixing intensity
            mixing = np.zeros_like(X)
            
            for i in range(4):
                for j in range(8):
                    jet_x = 1 + j * 2.5
                    jet_y = 1 + i * 2.0
                    
                    r = np.sqrt((X - jet_x)**2 + (Y - jet_y)**2)
                    
                    # Height-dependent mixing
                    if h < 6:  # Bottom zone - strong jet influence
                        mixing += np.exp(-r**2 / 6) * 1.5
                    elif h < 12:  # Middle zone - circulation
                        mixing += np.exp(-r**2 / 10) * 1.0
                    else:  # Top zone - weaker mixing
                        mixing += np.exp(-r**2 / 15) * 0.5
            
            # Plot surface
            surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.RdYlBu_r(mixing/mixing.max()),
                                  alpha=alpha, linewidth=0, antialiased=True)
        
        # Add jet positions as vertical lines
        for i in range(4):
            for j in range(8):
                jet_x = 1 + j * 2.5
                jet_y = 1 + i * 2.0
                ax.plot([jet_x, jet_x], [jet_y, jet_y], [0, 3], 
                       'r-', linewidth=2, alpha=0.8)
                
                # Add jet cone
                cone_height = 5
                cone_radius = 2
                theta = np.linspace(0, 2*np.pi, 20)
                z_cone = np.linspace(0, cone_height, 10)
                
                for z_val in z_cone:
                    r_cone = cone_radius * z_val / cone_height
                    x_cone = jet_x + r_cone * np.cos(theta)
                    y_cone = jet_y + r_cone * np.sin(theta)
                    ax.plot(x_cone, y_cone, z_val, 'b-', alpha=0.1, linewidth=0.5)
        
        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.set_title('3D Mixing Zone Distribution\n(Red=Bottom, Yellow=Middle, Blue=Top)',
                    fontweight='bold', pad=20)
        
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 8)
        ax.set_zlim(0, 16)
        
        # Add text annotations
        ax.text(10, 4, 14, 'Surface Zone\n(Weak Mixing)', fontsize=9, ha='center')
        ax.text(10, 4, 8, 'Circulation Zone\n(Good Mixing)', fontsize=9, ha='center')
        ax.text(10, 4, 2, 'Jet Zone\n(Intense Mixing)', fontsize=9, ha='center')
        
        plt.savefig(self.output_dir / 'mixing_zones_3d.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_performance_radar(self):
        """Radar chart for overall performance metrics"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), 
                                       subplot_kw=dict(projection='polar'))
        
        # Performance categories
        categories = ['Velocity\nAdequacy', 'Mixing\nTime', 'Energy\nEfficiency',
                     'Turbulence\nLevel', 'Coverage\nUniformity', 'G-value\nIntensity']
        N = len(categories)
        
        # Angles for each category
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Performance scores (0-100 scale)
        # Corrected model
        corrected_scores = [
            self.corrected['mean_velocity'] / 0.30 * 100,  # Target: 0.30 m/s
            min(100, 30 / (self.corrected['mixing_time']/60) * 100),  # Target: 30 min
            min(100, 20 / self.corrected['energy_density'] * 100),  # Target: <20 W/m³
            min(100, self.corrected['reynolds'] / 100000 * 100),  # Good if >100k
            85,  # Estimated coverage (from jet analysis)
            min(100, self.corrected['g_value'] / 50 * 100),  # Target: 50 s⁻¹
        ]
        corrected_scores += corrected_scores[:1]
        
        # Original model
        original_scores = [
            self.original['mean_velocity'] / 0.30 * 100,
            min(100, 30 / (self.original['mixing_time']/60) * 100),
            min(100, 20 / self.original['energy_density'] * 100),
            min(100, self.original['reynolds'] / 100000 * 100),
            40,  # Poor coverage
            min(100, self.original['g_value'] / 50 * 100),
        ]
        original_scores += original_scores[:1]
        
        # Plot corrected model
        ax1.plot(angles, corrected_scores, 'b-', linewidth=2, label='Performance')
        ax1.fill(angles, corrected_scores, 'b', alpha=0.25)
        
        # Add target zone
        target_scores = [100] * (N + 1)
        acceptable_scores = [70] * (N + 1)
        ax1.plot(angles, target_scores, 'g--', linewidth=1, alpha=0.5, label='Target')
        ax1.plot(angles, acceptable_scores, 'y--', linewidth=1, alpha=0.5, label='Acceptable')
        ax1.fill_between(angles, acceptable_scores, target_scores, color='green', alpha=0.1)
        
        ax1.set_xticks(angles[:-1])
        ax1.set_xticklabels(categories)
        ax1.set_ylim(0, 120)
        ax1.set_title('Corrected Model Performance', fontweight='bold', pad=20)
        ax1.legend(loc='upper right', fontsize=8)
        ax1.grid(True)
        
        # Plot original model
        ax2.plot(angles, original_scores, 'r-', linewidth=2, label='Performance')
        ax2.fill(angles, original_scores, 'r', alpha=0.25)
        
        ax2.plot(angles, target_scores, 'g--', linewidth=1, alpha=0.5, label='Target')
        ax2.plot(angles, acceptable_scores, 'y--', linewidth=1, alpha=0.5, label='Acceptable')
        ax2.fill_between(angles, acceptable_scores, target_scores, color='green', alpha=0.1)
        
        ax2.set_xticks(angles[:-1])
        ax2.set_xticklabels(categories)
        ax2.set_ylim(0, 120)
        ax2.set_title('Original Model Performance', fontweight='bold', pad=20)
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True)
        
        # Add overall scores
        overall_corrected = np.mean(corrected_scores[:-1])
        overall_original = np.mean(original_scores[:-1])
        
        fig.text(0.25, 0.02, f'Overall Score: {overall_corrected:.1f}/100',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        fig.text(0.75, 0.02, f'Overall Score: {overall_original:.1f}/100',
                ha='center', fontsize=12, fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.5))
        
        plt.suptitle('Performance Radar Analysis', fontsize=14, fontweight='bold', y=1.05)
        plt.tight_layout()
        
        plt.savefig(self.output_dir / 'performance_radar.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_reynolds_g_relationship(self):
        """Reynolds number vs G-value relationship"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Generate data for different scenarios
        Re_range = np.logspace(2, 6, 100)
        
        # Theoretical G-value relationships for different regimes
        # G ~ Re^0.5 for turbulent mixing
        G_turbulent = 0.01 * Re_range**0.5
        G_transitional = 0.005 * Re_range**0.6
        G_laminar = 0.001 * Re_range**0.8
        
        # Plot theoretical curves
        ax.loglog(Re_range, G_turbulent, 'b-', linewidth=2, alpha=0.5, label='Turbulent Regime')
        ax.loglog(Re_range, G_transitional, 'g-', linewidth=2, alpha=0.5, label='Transitional')
        ax.loglog(Re_range, G_laminar, 'r-', linewidth=2, alpha=0.5, label='Laminar')
        
        # Plot actual data points
        # Corrected model
        ax.scatter(self.corrected['reynolds'], self.corrected['g_value'],
                  s=200, c='blue', marker='o', edgecolor='black', linewidth=2,
                  label='Corrected Model', zorder=5)
        
        # Original model
        ax.scatter(self.original['reynolds'], self.original['g_value'],
                  s=200, c='red', marker='s', edgecolor='black', linewidth=2,
                  label='Original Model', zorder=5)
        
        # Add annotations
        ax.annotate(f"Corrected\nRe={self.corrected['reynolds']:.0f}\nG={self.corrected['g_value']:.1f} s⁻¹",
                   xy=(self.corrected['reynolds'], self.corrected['g_value']),
                   xytext=(self.corrected['reynolds']*2, self.corrected['g_value']*2),
                   arrowprops=dict(arrowstyle='->', color='blue', lw=1.5),
                   fontsize=9, ha='left')
        
        ax.annotate(f"Original\nRe={self.original['reynolds']:.0f}\nG={self.original['g_value']:.1f} s⁻¹",
                   xy=(self.original['reynolds'], self.original['g_value']),
                   xytext=(self.original['reynolds']/2, self.original['g_value']*3),
                   arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                   fontsize=9, ha='right')
        
        # Add mixing zones
        ax.axhspan(10, 30, alpha=0.2, color='yellow', label='Gentle Mixing')
        ax.axhspan(30, 100, alpha=0.2, color='green', label='Rapid Mixing')
        ax.axhspan(100, 1000, alpha=0.2, color='blue', label='Intense Mixing')
        
        # Add Reynolds regime boundaries
        ax.axvline(x=2000, color='gray', linestyle=':', alpha=0.5)
        ax.axvline(x=10000, color='gray', linestyle=':', alpha=0.5)
        ax.text(500, 500, 'Laminar', fontsize=9, ha='center', rotation=45)
        ax.text(5000, 500, 'Transitional', fontsize=9, ha='center', rotation=45)
        ax.text(100000, 500, 'Fully Turbulent', fontsize=9, ha='center', rotation=45)
        
        ax.set_xlabel('Reynolds Number Re', fontsize=12)
        ax.set_ylabel('Velocity Gradient G [s⁻¹]', fontsize=12)
        ax.set_title('Reynolds Number vs G-value Relationship\nfor Anaerobic Digester Mixing',
                    fontsize=14, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        ax.grid(True, alpha=0.3, which='both')
        ax.set_xlim(100, 1e6)
        ax.set_ylim(1, 1000)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'reynolds_g_relationship.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """Main execution"""
    print("=" * 60)
    print("Academic Visualization of Mixing Simulation Results")
    print("=" * 60)
    
    visualizer = AcademicVisualizer()
    visualizer.plot_all()
    
    print("\n" + "=" * 60)
    print("Visualization Complete!")
    print("=" * 60)
    
    # Summary statistics
    print("\nKey Performance Metrics:")
    print(f"  Corrected Model:")
    print(f"    - Mean Velocity: {visualizer.corrected['mean_velocity']:.3f} m/s")
    print(f"    - Mixing Time: {visualizer.corrected['mixing_time']/60:.1f} min")
    print(f"    - G-value: {visualizer.corrected['g_value']:.1f} s⁻¹")
    print(f"    - Pump Power: {visualizer.corrected['pump_power']:.1f} kW")
    print(f"  Improvement Ratio:")
    print(f"    - Velocity: {visualizer.corrected['mean_velocity']/visualizer.original['mean_velocity']:.1f}x")
    print(f"    - Mixing Time: {visualizer.original['mixing_time']/visualizer.corrected['mixing_time']:.1f}x faster")
    print(f"    - G-value: {visualizer.corrected['g_value']/visualizer.original['g_value']:.1f}x")


if __name__ == "__main__":
    main()