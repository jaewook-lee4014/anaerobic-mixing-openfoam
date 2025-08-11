#!/usr/bin/env python3
"""
3D Interactive Visualization for Anaerobic Digester Mixing System
브라우저에서 회전/확대 가능한 3D 시각화
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import webbrowser
from datetime import datetime

class Interactive3DVisualizer:
    """3D 인터랙티브 시각화 클래스"""
    
    def __init__(self):
        self.output_dir = Path("data/processed/3d_interactive")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 시스템 사양
        self.tank = {
            'L': 20.0,  # Length (m)
            'W': 8.0,   # Width (m) 
            'H': 16.0,  # Height (m)
            'volume': 2560.0  # m³
        }
        
        self.nozzles = {
            'count': 32,
            'diameter': 35.0,  # mm
            'angle': 45.0,  # degrees
            'velocity': 3.9,  # m/s
            'array': [4, 8],  # 4 rows × 8 columns
        }
        
        self.results = {
            'mean_velocity': 0.34,  # m/s
            'mixing_time': 28,  # minutes
            'g_value': 68,  # s⁻¹
        }
        
    def create_all_visualizations(self):
        """모든 3D 시각화 생성"""
        print("=" * 60)
        print("3D Interactive Visualization Generation")
        print("=" * 60)
        
        # 1. 탱크 및 노즐 배치 3D 시각화
        self.visualize_tank_and_nozzles()
        print("✓ Tank and nozzle configuration created")
        
        # 2. 속도장 3D 시각화
        self.visualize_velocity_field_3d()
        print("✓ 3D velocity field created")
        
        # 3. 혼합 영역 3D 시각화
        self.visualize_mixing_zones_3d()
        print("✓ 3D mixing zones created")
        
        # 4. 통합 대시보드
        self.create_integrated_dashboard()
        print("✓ Integrated dashboard created")
        
        print(f"\n✅ All visualizations saved to {self.output_dir}")
        print("📌 Open the HTML files in your browser to interact with the 3D models")
        
    def visualize_tank_and_nozzles(self):
        """탱크 및 노즐 배치 3D 시각화"""
        fig = go.Figure()
        
        # 1. 탱크 외곽선 (와이어프레임)
        # 바닥면
        x_bottom = [0, self.tank['L'], self.tank['L'], 0, 0]
        y_bottom = [0, 0, self.tank['W'], self.tank['W'], 0]
        z_bottom = [0, 0, 0, 0, 0]
        
        # 상단면
        x_top = [0, self.tank['L'], self.tank['L'], 0, 0]
        y_top = [0, 0, self.tank['W'], self.tank['W'], 0]
        z_top = [self.tank['H']] * 5
        
        # 탱크 와이어프레임
        for i in range(4):
            x_edge = [x_bottom[i], x_top[i]]
            y_edge = [y_bottom[i], y_top[i]]
            z_edge = [0, self.tank['H']]
            fig.add_trace(go.Scatter3d(
                x=x_edge, y=y_edge, z=z_edge,
                mode='lines',
                line=dict(color='gray', width=2),
                showlegend=False
            ))
        
        # 바닥과 상단 테두리
        fig.add_trace(go.Scatter3d(
            x=x_bottom, y=y_bottom, z=z_bottom,
            mode='lines',
            line=dict(color='gray', width=2),
            name='Tank',
            showlegend=True
        ))
        
        fig.add_trace(go.Scatter3d(
            x=x_top, y=y_top, z=z_top,
            mode='lines',
            line=dict(color='gray', width=2),
            showlegend=False
        ))
        
        # 2. 노즐 위치 및 제트 방향
        nozzle_positions = []
        jet_vectors = []
        
        x_spacing = self.tank['L'] / (self.nozzles['array'][1] + 1)
        y_spacing = self.tank['W'] / (self.nozzles['array'][0] + 1)
        
        for i in range(self.nozzles['array'][0]):
            for j in range(self.nozzles['array'][1]):
                x = (j + 1) * x_spacing
                y = (i + 1) * y_spacing
                z = 0.5  # 바닥 근처
                
                nozzle_positions.append([x, y, z])
                
                # 제트 방향 (45도 상향, 중앙 향)
                center_x = self.tank['L'] / 2
                center_y = self.tank['W'] / 2
                
                # 수평 방향 (중앙으로)
                dx_horizontal = center_x - x
                dy_horizontal = center_y - y
                distance = np.sqrt(dx_horizontal**2 + dy_horizontal**2)
                
                # 정규화
                if distance > 0:
                    dx_horizontal /= distance
                    dy_horizontal /= distance
                
                # 45도 상향 적용
                jet_length = 3.0
                dx = dx_horizontal * jet_length * np.cos(np.radians(45))
                dy = dy_horizontal * jet_length * np.cos(np.radians(45))
                dz = jet_length * np.sin(np.radians(45))
                
                jet_vectors.append([dx, dy, dz])
        
        # 노즐 포인트
        nozzle_positions = np.array(nozzle_positions)
        fig.add_trace(go.Scatter3d(
            x=nozzle_positions[:, 0],
            y=nozzle_positions[:, 1],
            z=nozzle_positions[:, 2],
            mode='markers',
            marker=dict(size=8, color='red', symbol='diamond'),
            name='Nozzles (32 EA)',
            text=[f'Nozzle {i+1}<br>Flow: 13.4 m³/h<br>Velocity: 3.9 m/s' 
                  for i in range(len(nozzle_positions))],
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # 제트 방향 화살표 (Cone으로 표현)
        jet_vectors = np.array(jet_vectors)
        for i, (pos, vec) in enumerate(zip(nozzle_positions, jet_vectors)):
            # 제트 코어 영역
            t = np.linspace(0, 1, 20)
            jet_x = pos[0] + vec[0] * t
            jet_y = pos[1] + vec[1] * t
            jet_z = pos[2] + vec[2] * t
            
            fig.add_trace(go.Scatter3d(
                x=jet_x, y=jet_y, z=jet_z,
                mode='lines',
                line=dict(color='blue', width=3),
                showlegend=False,
                opacity=0.6
            ))
            
            # 제트 확산 영역 (원뿔)
            if i < 4:  # 일부만 표시 (성능 고려)
                theta = np.linspace(0, 2*np.pi, 10)
                for t_val in [0.3, 0.6, 0.9]:
                    radius = t_val * 0.5
                    circle_x = pos[0] + vec[0] * t_val + radius * np.cos(theta) * vec[1] / 3
                    circle_y = pos[1] + vec[1] * t_val + radius * np.sin(theta) * vec[0] / 3
                    circle_z = pos[2] + vec[2] * t_val * np.ones_like(theta)
                    
                    fig.add_trace(go.Scatter3d(
                        x=circle_x, y=circle_y, z=circle_z,
                        mode='lines',
                        line=dict(color='cyan', width=1),
                        showlegend=False,
                        opacity=0.3
                    ))
        
        # 3. 중앙 스크린
        screen_center = [self.tank['L']/2, self.tank['W']/2, self.tank['H']/2]
        screen_radius = 0.3
        screen_height = 1.0
        
        # 스크린 실린더
        theta = np.linspace(0, 2*np.pi, 20)
        z_screen = np.linspace(screen_center[2] - screen_height/2, 
                               screen_center[2] + screen_height/2, 10)
        
        for z in z_screen:
            screen_x = screen_center[0] + screen_radius * np.cos(theta)
            screen_y = screen_center[1] + screen_radius * np.sin(theta)
            screen_z = z * np.ones_like(theta)
            
            fig.add_trace(go.Scatter3d(
                x=screen_x, y=screen_y, z=screen_z,
                mode='lines',
                line=dict(color='green', width=1),
                showlegend=False,
                opacity=0.5
            ))
        
        # 레이아웃 설정
        fig.update_layout(
            title=dict(
                text=f'3D Tank Configuration<br>Volume: {self.tank["volume"]:.0f} m³ | '
                     f'32 Nozzles @ 45° upward',
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(title='Length X [m]', range=[0, self.tank['L']]),
                yaxis=dict(title='Width Y [m]', range=[0, self.tank['W']]),
                zaxis=dict(title='Height Z [m]', range=[0, self.tank['H']]),
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectmode='data'
            ),
            height=800,
            showlegend=True,
            hovermode='closest'
        )
        
        # HTML 저장
        output_file = self.output_dir / 'tank_nozzle_3d.html'
        fig.write_html(str(output_file))
        print(f"  → Saved: {output_file}")
        
    def visualize_velocity_field_3d(self):
        """3D 속도장 시각화"""
        fig = go.Figure()
        
        # 속도장 그리드 생성
        nx, ny, nz = 20, 10, 16
        x = np.linspace(0, self.tank['L'], nx)
        y = np.linspace(0, self.tank['W'], ny)
        z = np.linspace(0, self.tank['H'], nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 속도장 모델링 (간단한 순환 패턴)
        center_x = self.tank['L'] / 2
        center_y = self.tank['W'] / 2
        
        # 거리 계산
        R = np.sqrt((X - center_x)**2 + (Y - center_y)**2)
        
        # 속도 성분
        U = np.zeros_like(X)
        V = np.zeros_like(X)
        W = np.zeros_like(X)
        
        # 상승/하강 순환 패턴
        # 중앙 상승
        upflow_mask = R < 5
        W[upflow_mask] = 0.4 * (1 - Z[upflow_mask]/self.tank['H']) * (1 - R[upflow_mask]/5)
        
        # 벽면 하강
        downflow_mask = R >= 5
        W[downflow_mask] = -0.2 * (Z[downflow_mask]/self.tank['H'])
        
        # 수평 순환
        U = -0.2 * (Y - center_y) / (R + 0.1) * np.exp(-Z/10)
        V = 0.2 * (X - center_x) / (R + 0.1) * np.exp(-Z/10)
        
        # 제트 영향 추가 (바닥 근처)
        jet_zone = Z < 4
        U[jet_zone] *= 2
        V[jet_zone] *= 2
        W[jet_zone] += 0.3
        
        # 속도 크기
        velocity_magnitude = np.sqrt(U**2 + V**2 + W**2)
        
        # 1. Isosurface (등속도면)
        fig.add_trace(go.Isosurface(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=velocity_magnitude.flatten(),
            isomin=0.1,
            isomax=0.5,
            surface_count=5,
            colorscale='RdYlBu_r',
            colorbar=dict(title='|V| [m/s]', x=1.1),
            caps=dict(x=dict(show=False), y=dict(show=False), z=dict(show=False)),
            opacity=0.3,
            name='Velocity Field'
        ))
        
        # 2. Streamlines (유선)
        # 시작점 설정
        seed_points = []
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    seed_points.append([
                        5 + i * 5,
                        2 + j * 2,
                        2 + k * 5
                    ])
        
        # 간단한 유선 추적
        for seed in seed_points[:9]:  # 성능을 위해 일부만
            streamline_x = [seed[0]]
            streamline_y = [seed[1]]
            streamline_z = [seed[2]]
            
            current_pos = seed.copy()
            for _ in range(30):
                # 현재 위치에서 속도 보간
                ix = int(current_pos[0] * nx / self.tank['L'])
                iy = int(current_pos[1] * ny / self.tank['W'])
                iz = int(current_pos[2] * nz / self.tank['H'])
                
                if 0 <= ix < nx and 0 <= iy < ny and 0 <= iz < nz:
                    dt = 0.5
                    current_pos[0] += U[ix, iy, iz] * dt
                    current_pos[1] += V[ix, iy, iz] * dt
                    current_pos[2] += W[ix, iy, iz] * dt
                    
                    # 경계 체크
                    current_pos[0] = np.clip(current_pos[0], 0, self.tank['L'])
                    current_pos[1] = np.clip(current_pos[1], 0, self.tank['W'])
                    current_pos[2] = np.clip(current_pos[2], 0, self.tank['H'])
                    
                    streamline_x.append(current_pos[0])
                    streamline_y.append(current_pos[1])
                    streamline_z.append(current_pos[2])
                else:
                    break
            
            # 유선 그리기
            fig.add_trace(go.Scatter3d(
                x=streamline_x,
                y=streamline_y,
                z=streamline_z,
                mode='lines',
                line=dict(
                    color=np.linspace(0, 1, len(streamline_x)),
                    colorscale='Viridis',
                    width=3
                ),
                showlegend=False,
                hovertemplate='Streamline<extra></extra>'
            ))
        
        # 3. 벡터 필드 (화살표)
        # 샘플링 (성능 고려)
        skip = 3
        x_sample = X[::skip, ::skip, ::skip].flatten()
        y_sample = Y[::skip, ::skip, ::skip].flatten()
        z_sample = Z[::skip, ::skip, ::skip].flatten()
        u_sample = U[::skip, ::skip, ::skip].flatten()
        v_sample = V[::skip, ::skip, ::skip].flatten()
        w_sample = W[::skip, ::skip, ::skip].flatten()
        
        # 속도 크기로 색상 결정
        vel_mag_sample = np.sqrt(u_sample**2 + v_sample**2 + w_sample**2)
        
        fig.add_trace(go.Cone(
            x=x_sample,
            y=y_sample,
            z=z_sample,
            u=u_sample,
            v=v_sample,
            w=w_sample,
            colorscale='Blues',
            sizemode='scaled',
            sizeref=0.5,
            showscale=False,
            opacity=0.6,
            name='Velocity Vectors'
        ))
        
        # 4. Dead Zone 표시 (|V| < 0.1 m/s)
        dead_zone_mask = velocity_magnitude < 0.1
        if np.any(dead_zone_mask):
            fig.add_trace(go.Isosurface(
                x=X.flatten(),
                y=Y.flatten(),
                z=Z.flatten(),
                value=dead_zone_mask.flatten().astype(float),
                isomin=0.5,
                isomax=1.0,
                surface_count=1,
                colorscale=[[0, 'red'], [1, 'red']],
                showscale=False,
                opacity=0.2,
                name='Dead Zones'
            ))
        
        # 레이아웃 설정
        fig.update_layout(
            title=dict(
                text=f'3D Velocity Field<br>Mean Velocity: {self.results["mean_velocity"]:.2f} m/s | '
                     f'Target: ≥0.30 m/s',
                font=dict(size=16)
            ),
            scene=dict(
                xaxis=dict(title='Length X [m]', range=[0, self.tank['L']]),
                yaxis=dict(title='Width Y [m]', range=[0, self.tank['W']]),
                zaxis=dict(title='Height Z [m]', range=[0, self.tank['H']]),
                camera=dict(
                    eye=dict(x=1.8, y=1.8, z=1.5)
                ),
                aspectmode='data'
            ),
            height=800,
            showlegend=True
        )
        
        # HTML 저장
        output_file = self.output_dir / 'velocity_field_3d.html'
        fig.write_html(str(output_file))
        print(f"  → Saved: {output_file}")
        
    def visualize_mixing_zones_3d(self):
        """3D 혼합 영역 시각화"""
        fig = go.Figure()
        
        # 그리드 생성
        nx, ny, nz = 30, 15, 20
        x = np.linspace(0, self.tank['L'], nx)
        y = np.linspace(0, self.tank['W'], ny)
        z = np.linspace(0, self.tank['H'], nz)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        
        # 혼합 강도 계산
        mixing_intensity = np.zeros_like(X)
        
        # 노즐 위치에서의 혼합 강도
        x_spacing = self.tank['L'] / (self.nozzles['array'][1] + 1)
        y_spacing = self.tank['W'] / (self.nozzles['array'][0] + 1)
        
        for i in range(self.nozzles['array'][0]):
            for j in range(self.nozzles['array'][1]):
                nozzle_x = (j + 1) * x_spacing
                nozzle_y = (i + 1) * y_spacing
                nozzle_z = 0.5
                
                # 거리 기반 혼합 강도
                R = np.sqrt((X - nozzle_x)**2 + (Y - nozzle_y)**2 + (Z - nozzle_z)**2)
                
                # 제트 영향 (높이에 따라 감소)
                jet_influence = np.exp(-R**2 / 10) * np.exp(-Z / 8)
                mixing_intensity += jet_influence
        
        # 정규화
        mixing_intensity = mixing_intensity / np.max(mixing_intensity)
        
        # 1. 혼합 영역 분류
        excellent_mask = mixing_intensity > 0.7
        good_mask = (mixing_intensity > 0.5) & (mixing_intensity <= 0.7)
        moderate_mask = (mixing_intensity > 0.3) & (mixing_intensity <= 0.5)
        poor_mask = (mixing_intensity > 0.1) & (mixing_intensity <= 0.3)
        dead_mask = mixing_intensity <= 0.1
        
        # 2. 각 영역을 Isosurface로 표현
        zones = [
            (excellent_mask, 'Excellent (>70%)', 'green', 0.5),
            (good_mask, 'Good (50-70%)', 'blue', 0.4),
            (moderate_mask, 'Moderate (30-50%)', 'yellow', 0.3),
            (poor_mask, 'Poor (10-30%)', 'orange', 0.3),
            (dead_mask, 'Dead Zone (<10%)', 'red', 0.2)
        ]
        
        for mask, name, color, opacity in zones:
            if np.any(mask):
                fig.add_trace(go.Isosurface(
                    x=X.flatten(),
                    y=Y.flatten(),
                    z=Z.flatten(),
                    value=mask.flatten().astype(float),
                    isomin=0.5,
                    isomax=1.0,
                    surface_count=1,
                    colorscale=[[0, color], [1, color]],
                    showscale=False,
                    opacity=opacity,
                    name=name,
                    hovertemplate=f'{name}<extra></extra>'
                ))
        
        # 3. 수평 단면 추가 (z/H = 0.1, 0.5, 0.9)
        slice_heights = [0.1, 0.5, 0.9]
        for h_ratio in slice_heights:
            z_slice = h_ratio * self.tank['H']
            iz = int(z_slice * nz / self.tank['H'])
            
            if 0 <= iz < nz:
                fig.add_trace(go.Surface(
                    x=x,
                    y=y,
                    z=np.ones((ny, nx)) * z_slice,
                    surfacecolor=mixing_intensity[:, :, iz].T,
                    colorscale='RdYlGn',
                    showscale=False,
                    opacity=0.6,
                    name=f'Slice at z/H={h_ratio:.1f}',
                    hovertemplate='z/H=' + f'{h_ratio:.1f}' + '<br>Mixing: %{surfacecolor:.2f}<extra></extra>'
                ))
        
        # 4. 통계 정보 추가 (텍스트 주석)
        total_volume = self.tank['volume']
        excellent_pct = np.sum(excellent_mask) / excellent_mask.size * 100
        good_pct = np.sum(good_mask) / good_mask.size * 100
        moderate_pct = np.sum(moderate_mask) / moderate_mask.size * 100
        poor_pct = np.sum(poor_mask) / poor_mask.size * 100
        dead_pct = np.sum(dead_mask) / dead_mask.size * 100
        
        stats_text = (
            f"Volume Distribution:<br>"
            f"Excellent: {excellent_pct:.1f}%<br>"
            f"Good: {good_pct:.1f}%<br>"
            f"Moderate: {moderate_pct:.1f}%<br>"
            f"Poor: {poor_pct:.1f}%<br>"
            f"Dead Zone: {dead_pct:.1f}%"
        )
        
        # 레이아웃 설정
        fig.update_layout(
            title=dict(
                text=f'3D Mixing Zones Analysis<br>{stats_text}',
                font=dict(size=14)
            ),
            scene=dict(
                xaxis=dict(title='Length X [m]', range=[0, self.tank['L']]),
                yaxis=dict(title='Width Y [m]', range=[0, self.tank['W']]),
                zaxis=dict(title='Height Z [m]', range=[0, self.tank['H']]),
                camera=dict(
                    eye=dict(x=1.5, y=-1.5, z=1.2),
                    center=dict(x=0, y=0, z=0)
                ),
                aspectmode='data'
            ),
            height=800,
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        # HTML 저장
        output_file = self.output_dir / 'mixing_zones_3d.html'
        fig.write_html(str(output_file))
        print(f"  → Saved: {output_file}")
        
    def create_integrated_dashboard(self):
        """통합 3D 대시보드"""
        # Subplot 생성
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{'type': 'scatter3d'}, {'type': 'scatter3d'}],
                [{'type': 'scatter3d'}, {'type': 'scatter'}]
            ],
            subplot_titles=(
                'Tank & Nozzle Configuration',
                'Velocity Field Streamlines',
                'Mixing Zone Distribution',
                'Performance Metrics'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # 1. 탱크 및 노즐 (좌상단)
        x_spacing = self.tank['L'] / (self.nozzles['array'][1] + 1)
        y_spacing = self.tank['W'] / (self.nozzles['array'][0] + 1)
        
        nozzle_x = []
        nozzle_y = []
        nozzle_z = []
        
        for i in range(self.nozzles['array'][0]):
            for j in range(self.nozzles['array'][1]):
                nozzle_x.append((j + 1) * x_spacing)
                nozzle_y.append((i + 1) * y_spacing)
                nozzle_z.append(0.5)
        
        fig.add_trace(
            go.Scatter3d(
                x=nozzle_x, y=nozzle_y, z=nozzle_z,
                mode='markers',
                marker=dict(size=6, color='red'),
                name='Nozzles',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 탱크 외곽선
        tank_edges_x = [0, self.tank['L'], self.tank['L'], 0, 0, 0, self.tank['L'], self.tank['L'], 0, 0]
        tank_edges_y = [0, 0, self.tank['W'], self.tank['W'], 0, 0, 0, self.tank['W'], self.tank['W'], 0]
        tank_edges_z = [0, 0, 0, 0, 0, self.tank['H'], self.tank['H'], self.tank['H'], self.tank['H'], self.tank['H']]
        
        fig.add_trace(
            go.Scatter3d(
                x=tank_edges_x, y=tank_edges_y, z=tank_edges_z,
                mode='lines',
                line=dict(color='gray', width=2),
                name='Tank',
                showlegend=False
            ),
            row=1, col=1
        )
        
        # 2. 속도장 유선 (우상단)
        # 간단한 유선들
        for i in range(5):
            t = np.linspace(0, 10, 50)
            x_stream = 10 + 5 * np.cos(t/2 + i)
            y_stream = 4 + 3 * np.sin(t/2 + i)
            z_stream = t * 1.5
            
            fig.add_trace(
                go.Scatter3d(
                    x=x_stream, y=y_stream, z=z_stream,
                    mode='lines',
                    line=dict(
                        color=t,
                        colorscale='Viridis',
                        width=3
                    ),
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # 3. 혼합 영역 (좌하단)
        # 간단한 구체들로 표현
        theta = np.linspace(0, 2*np.pi, 20)
        phi = np.linspace(0, np.pi, 20)
        
        for zone_center, zone_radius, zone_color in [
            ([10, 4, 8], 3, 'green'),
            ([5, 2, 4], 2, 'yellow'),
            ([15, 6, 12], 2.5, 'orange')
        ]:
            x_sphere = zone_center[0] + zone_radius * np.outer(np.cos(theta), np.sin(phi))
            y_sphere = zone_center[1] + zone_radius * np.outer(np.sin(theta), np.sin(phi))
            z_sphere = zone_center[2] + zone_radius * np.outer(np.ones(np.size(theta)), np.cos(phi))
            
            fig.add_trace(
                go.Surface(
                    x=x_sphere, y=y_sphere, z=z_sphere,
                    colorscale=[[0, zone_color], [1, zone_color]],
                    showscale=False,
                    opacity=0.3,
                    showlegend=False
                ),
                row=2, col=1
            )
        
        # 4. 성능 메트릭 (우하단)
        metrics = ['Velocity', 'Mixing Time', 'MLSS Dev', 'Energy', 'G-value']
        actual = [0.34/0.30*100, (30-28)/30*100, (5-4.7)/5*100, 4.9/5*100, 68/65*100]
        target = [100, 100, 100, 100, 100]
        
        fig.add_trace(
            go.Bar(
                x=metrics,
                y=actual,
                name='Actual',
                marker_color='green',
                showlegend=True
            ),
            row=2, col=2
        )
        
        fig.add_trace(
            go.Scatter(
                x=metrics,
                y=target,
                mode='lines',
                name='Target',
                line=dict(color='red', dash='dash'),
                showlegend=True
            ),
            row=2, col=2
        )
        
        # 레이아웃 업데이트
        fig.update_layout(
            title=dict(
                text=f'Anaerobic Digester 3D Analysis Dashboard<br>'
                     f'<sub>Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</sub>',
                font=dict(size=18)
            ),
            height=900,
            showlegend=True,
            scene=dict(
                aspectmode='data',
                xaxis=dict(title='X [m]'),
                yaxis=dict(title='Y [m]'),
                zaxis=dict(title='Z [m]')
            ),
            scene2=dict(
                aspectmode='data',
                xaxis=dict(title='X [m]'),
                yaxis=dict(title='Y [m]'),
                zaxis=dict(title='Z [m]')
            ),
            scene3=dict(
                aspectmode='data',
                xaxis=dict(title='X [m]'),
                yaxis=dict(title='Y [m]'),
                zaxis=dict(title='Z [m]')
            )
        )
        
        # HTML 저장
        output_file = self.output_dir / 'integrated_dashboard_3d.html'
        fig.write_html(str(output_file))
        print(f"  → Saved: {output_file}")
        
        # 자동으로 브라우저 열기
        print(f"\n📌 Opening dashboard in browser...")
        webbrowser.open(f'file://{output_file.absolute()}')


def main():
    """메인 실행"""
    visualizer = Interactive3DVisualizer()
    visualizer.create_all_visualizations()
    
    print("\n" + "=" * 60)
    print("✅ 3D Interactive Visualization Complete!")
    print("=" * 60)
    print("\nTo view the 3D models:")
    print("1. Open the HTML files in your browser")
    print("2. Use mouse to rotate, scroll to zoom")
    print("3. Click on legend items to show/hide elements")
    print("\nFiles location:", visualizer.output_dir)


if __name__ == "__main__":
    main()