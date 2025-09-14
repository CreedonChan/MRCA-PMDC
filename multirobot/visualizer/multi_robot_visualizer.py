import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, PathPatch, Polygon
from matplotlib.path import Path
from typing import Dict, List, Tuple, Optional, Union
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter, PillowWriter
from math import cos, sin
import copy

class MultiRobotVisualizer:
    def __init__(self, workspace_limits: Union[List[List[float]], Tuple[Tuple[float]]],
                 show_display=True, show_ids: bool = True, show_obstacle_ids: bool = True):
        self.show_display = show_display
        self.show_ids = show_ids
        self.show_obstacle_ids = show_obstacle_ids

        self.fig, self.ax = plt.subplots(figsize=(4, 4))

        self.workspace_limits = np.array(workspace_limits)
        self.ax.set_xlim(self.workspace_limits[0][0], self.workspace_limits[0][1])
        self.ax.set_ylim(self.workspace_limits[1][0], self.workspace_limits[1][1])

        self.ax.set_aspect('equal')
        self.ax.grid(True)

        self.ax.set_xlabel('x (m)')
        self.ax.set_ylabel('y (m)')

        self.robot_artists = {}
        self.obstacle_artists = {}
        self.rectangle_obstacle_artists = {}

        self.time_text = self.ax.text(
            0.02, 0.95, '',
            transform=self.ax.transAxes,
            fontsize=10,
            bbox=dict(facecolor='white', alpha=0.8)
        )

        self.robot_colors = list(mcolors.TABLEAU_COLORS.values())

        self.obstacle_color = '#808080'
        self.obstacle_alpha = 0.7
        self.obstacle_edge_color = '#404040'
        self.obstacle_edge_width = 1.5

        self.target_marker_size = 0.2

        self.writer = None
        self.frame_count = 0
        self.frames_for_gif = []

        self.initial_robot_states = {}
        self.initial_obstacle_states = {}

        self.final_robot_positions = {}

    def setup_video_writer(self, filename: str = 'output.mp4', fps: int = 30, dpi: int = 100):
        if filename.endswith('.gif'):
            self.writer = PillowWriter(fps=fps)
        else:
            self.writer = FFMpegWriter(fps=fps)
        self.writer.setup(self.fig, filename, dpi=dpi)
        self.frame_count = 0

    def setup_gif_recording(self):
        self.frames_for_gif = []

    def save_gif(self, filename: str = 'output.gif', fps: int = 20, dpi: int = 100):
        if not self.frames_for_gif:
            print("No frame data available")
            return

        print(f"Saving GIF animation to {filename}...")

        writer = PillowWriter(fps=fps)
        writer.setup(self.fig, filename, dpi=dpi)

        for frame in self.frames_for_gif:
            writer.grab_frame()

        writer.finish()
        print(f"GIF saved, total {len(self.frames_for_gif)} frames")
        self.frames_for_gif = []

    def _get_robot_color(self, robot_id: int) -> Tuple[float, float, float]:
        return mcolors.to_rgb(self.robot_colors[robot_id % len(self.robot_colors)])

    def add_robot(self,
                  robot_id: int,
                  init_pos: np.ndarray,
                  radius: float = 0.2,
                  color: Optional[Tuple[float, float, float]] = None,
                  show_orientation: bool = True,
                  show_trajectory: bool = True,
                  show_id: Optional[bool] = None):
        color = color or self._get_robot_color(robot_id)

        if show_id is None:
            show_id = self.show_ids

        body = Circle(init_pos.tolist(), radius, facecolor=color, edgecolor='k', alpha=0.8, zorder=10)
        self.ax.add_patch(body)

        arrow = None
        if show_orientation:
            arrow = PathPatch(
                Path([(0, 0), (1, 0)], [Path.MOVETO, Path.LINETO]),
                facecolor='none', edgecolor=color, linewidth=2, zorder=11
            )
            self.ax.add_patch(arrow)

        traj_line = None
        if show_trajectory:
            traj_line, = self.ax.plot([], [], color=color, linewidth=1.5, alpha=0.6, zorder=5)

        id_text = None
        if show_id:
            id_text = self.ax.text(
                init_pos[0], init_pos[1] + radius * 1.5,
                f"ID: {robot_id}",
                color=color,
                ha='center', va='bottom',
                fontsize=13,
                fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1),
                zorder=12
            )

        self.initial_robot_states[robot_id] = {
            'pos_real': init_pos.copy(),
            'orientation': 0.0,
            'radius': radius,
            'color': color,
            'show_orientation': show_orientation,
            'show_trajectory': show_trajectory,
            'show_id': show_id
        }

        self.final_robot_positions[robot_id] = init_pos.copy()

        self.robot_artists[robot_id] = {
            'body': body,
            'arrow': arrow,
            'trajectory': traj_line,
            'id_text': id_text,
            'traj_data': [init_pos.copy()] if show_trajectory else None,
            'radius': radius,
            'show_orientation': show_orientation,
            'show_id': show_id,
            'color': color,
            'current_pos': init_pos.copy()
        }

    def add_obstacle(self,
                     obs_id: int,
                     position: np.ndarray,
                     radius: float = 0.5,
                     show_id: Optional[bool] = None):
        if show_id is None:
            show_id = self.show_obstacle_ids

        obs = Circle(
            position.tolist(), radius,
            facecolor=self.obstacle_color,
            edgecolor=self.obstacle_edge_color,
            alpha=self.obstacle_alpha,
            linewidth=self.obstacle_edge_width,
            zorder=1
        )
        self.ax.add_patch(obs)

        id_text = None
        if show_id:
            id_text = self.ax.text(
                position[0], position[1],
                str(obs_id),
                color='white',
                ha='center', va='center',
                fontsize=13,
                fontweight='bold',
                zorder=2
            )

        self.initial_obstacle_states[obs_id] = {
            'pos_real': position.copy(),
            'radius': radius,
            'color': self.obstacle_color,
            'show_id': show_id,
            'obs_type': 'circle_obs'
        }

        self.obstacle_artists[obs_id] = {
            'patch': obs,
            'id_text': id_text,
            'color': self.obstacle_color,
            'type': 'circle',
            'show_id': show_id,
            'initial_pos': position.copy(),
            'current_pos': position.copy()
        }

    def add_rectangle_obstacle(self,
                               obs_id: int,
                               center: np.ndarray,
                               size: Tuple[float, float],
                               yaw: float = 0.0,
                               show_id: Optional[bool] = None):
        if show_id is None:
            show_id = self.show_obstacle_ids

        half_length, half_width = size[0] / 2, size[1] / 2
        vertices = np.array([
            [-half_length, -half_width],
            [half_length, -half_width],
            [half_length, half_width],
            [-half_length, half_width]
        ])

        rot_mat = np.array([
            [cos(yaw), -sin(yaw)],
            [sin(yaw), cos(yaw)]
        ])

        rotated_vertices = np.dot(vertices, rot_mat.T) + center

        rect = Polygon(
            rotated_vertices,
            closed=True,
            facecolor=self.obstacle_color,
            edgecolor=self.obstacle_edge_color,
            alpha=self.obstacle_alpha,
            linewidth=self.obstacle_edge_width,
            zorder=1
        )
        self.ax.add_patch(rect)

        id_text = None
        if show_id:
            id_text = self.ax.text(
                center[0], center[1],
                str(obs_id),
                color='white',
                ha='center', va='center',
                fontsize=13,
                fontweight='bold',
                zorder=2
            )

        self.initial_obstacle_states[obs_id] = {
            'pos_real': center.copy(),
            'size': size,
            'yaw': yaw,
            'color': self.obstacle_color,
            'show_id': show_id,
            'obs_type': 'rectangle_obs'
        }

        self.rectangle_obstacle_artists[obs_id] = {
            'patch': rect,
            'id_text': id_text,
            'initial_pos': center.copy(),
            'initial_size': size,
            'initial_yaw': yaw,
            'current_pos': center.copy(),
            'current_size': size,
            'current_yaw': yaw,
            'color': self.obstacle_color,
            'type': 'rectangle',
            'show_id': show_id
        }

    def update(self,
               robot_states: Dict[int, Dict],
               obstacle_states: Dict[int, Dict],
               time: float):
        for robot_id, state in robot_states.items():
            if robot_id not in self.robot_artists:
                continue

            artist = self.robot_artists[robot_id]
            pos = state['pos_real']
            artist['body'].center = pos
            artist['current_pos'] = pos.copy()

            self.final_robot_positions[robot_id] = pos.copy()

            if artist['arrow'] is not None:
                if state.get('show_orientation', artist['show_orientation']):
                    orientation = state['orientation']
                    arrow_len = state.get('radius', artist['radius']) * 1.5
                    dx = arrow_len * np.cos(orientation)
                    dy = arrow_len * np.sin(orientation)
                    artist['arrow'].set_path(
                        Path([pos, pos + [dx, dy]], [Path.MOVETO, Path.LINETO])
                    )
                    artist['arrow'].set_visible(True)
                else:
                    artist['arrow'].set_visible(False)

            if artist['trajectory'] is not None:
                if state.get('show_trajectory', True):
                    artist['traj_data'].append(pos.copy())
                    traj = np.array(artist['traj_data'])
                    artist['trajectory'].set_data(traj[:, 0], traj[:, 1])
                    artist['trajectory'].set_visible(True)
                else:
                    artist['trajectory'].set_visible(False)

            if artist['id_text'] is not None:
                show_id = state.get('show_id', artist['show_id'])
                artist['id_text'].set_visible(show_id)
                if show_id:
                    artist['id_text'].set_position((pos[0], pos[1] + artist['radius'] * 1.5))
                    if 'id_text' in state:
                        artist['id_text'].set_text(state['id_text'])
                    else:
                        artist['id_text'].set_text(f"ID: {robot_id}")

            if 'radius' in state:
                artist['body'].radius = state['radius']
                artist['radius'] = state['radius']

            if 'color' in state and state['color'] is not None:
                new_color = state['color']
                artist['body'].set_facecolor(new_color)
                if artist['arrow'] is not None:
                    artist['arrow'].set_edgecolor(new_color)
                if artist['trajectory'] is not None:
                    artist['trajectory'].set_color(new_color)
                if artist['id_text'] is not None:
                    artist['id_text'].set_color(new_color)
                artist['color'] = new_color

        for obs_id, state in obstacle_states.items():
            if state.get('obs_type') == 'circle_obs':
                if obs_id in self.obstacle_artists:
                    artist = self.obstacle_artists[obs_id]
                    pos = state['pos_real']
                    artist['patch'].center = pos
                    artist['current_pos'] = pos.copy()

                    if artist['id_text'] is not None:
                        show_id = state.get('show_id', artist['show_id'])
                        artist['id_text'].set_visible(show_id)
                        if show_id:
                            artist['id_text'].set_position((pos[0], pos[1]))

            elif state.get('obs_type') == 'rectangle_obs':
                if obs_id not in self.rectangle_obstacle_artists:
                    continue

                artist = self.rectangle_obstacle_artists[obs_id]
                pos_real = state.get('pos_real', artist['current_pos'])
                yaw = state.get('yaw', artist['current_yaw'])
                size = state.get('size', artist['current_size'])

                artist['current_pos'] = pos_real.copy()
                artist['current_yaw'] = yaw
                artist['current_size'] = size

                half_length, half_width = size[0] / 2, size[1] / 2
                vertices = np.array([
                    [-half_length, -half_width],
                    [half_length, -half_width],
                    [half_length, half_width],
                    [-half_length, half_width]
                ])

                rot_mat = np.array([
                    [cos(yaw), -sin(yaw)],
                    [sin(yaw), cos(yaw)]
                ])

                rotated_vertices = np.dot(vertices, rot_mat.T) + pos_real
                artist['patch'].set_xy(rotated_vertices)

                if artist['id_text'] is not None:
                    show_id = state.get('show_id', artist['show_id'])
                    artist['id_text'].set_visible(show_id)
                    if show_id:
                        artist['id_text'].set_position((pos_real[0], pos_real[1]))

        self.time_text.set_text(f'Time: {time:.1f}s')

        if self.show_display:
            plt.draw()
            plt.pause(0.01)

        if self.writer is not None:
            self.writer.grab_frame()
            self.frame_count += 1

        if hasattr(self, 'frames_for_gif') and self.frames_for_gif is not None:
            self.fig.canvas.draw()
            frame = np.array(self.fig.canvas.renderer.buffer_rgba())
            self.frames_for_gif.append(frame)

    def close_video_writer(self):
        if self.writer is not None:
            self.writer.finish()
            print(f"Video saved, total {self.frame_count} frames")
            self.writer = None

    def _save_frame_as_pdf(self, use_current_positions: bool, filename: str, include_trajectories: bool = False,
                           show_targets: bool = False):
        print(f"Saving frame to {filename}...")

        save_fig, save_ax = plt.subplots(figsize=self.fig.get_size_inches())

        for obs_id, obs_data in self.obstacle_artists.items():
            if obs_data['type'] == 'circle':
                if use_current_positions:
                    pos = obs_data['current_pos']
                else:
                    pos = obs_data['initial_pos']

                radius = obs_data['patch'].radius
                show_id = obs_data['show_id']

                new_patch = Circle(
                    pos, radius,
                    facecolor=self.obstacle_color,
                    edgecolor=self.obstacle_edge_color,
                    alpha=self.obstacle_alpha,
                    linewidth=self.obstacle_edge_width,
                    zorder=1
                )
                save_ax.add_patch(new_patch)

                if show_id:
                    save_ax.text(
                        pos[0], pos[1],
                        str(obs_id),
                        color='white',
                        ha='center', va='center',
                        fontsize=13,
                        fontweight='bold',
                        zorder=2
                    )

        for obs_id, obs_data in self.rectangle_obstacle_artists.items():
            if obs_data['type'] == 'rectangle':
                if use_current_positions:
                    pos_real = obs_data['current_pos']
                    size = obs_data['current_size']
                    yaw = obs_data['current_yaw']
                else:
                    pos_real = obs_data['initial_pos']
                    size = obs_data['initial_size']
                    yaw = obs_data['initial_yaw']

                show_id = obs_data['show_id']

                half_length, half_width = size[0] / 2, size[1] / 2
                vertices = np.array([
                    [-half_length, -half_width],
                    [half_length, -half_width],
                    [half_length, half_width],
                    [-half_length, half_width]
                ])

                rot_mat = np.array([
                    [cos(yaw), -sin(yaw)],
                    [sin(yaw), cos(yaw)]
                ])

                rotated_vertices = np.dot(vertices, rot_mat.T) + pos_real

                new_patch = Polygon(
                    rotated_vertices,
                    closed=True,
                    facecolor=self.obstacle_color,
                    edgecolor=self.obstacle_edge_color,
                    alpha=self.obstacle_alpha,
                    linewidth=self.obstacle_edge_width,
                    zorder=1
                )
                save_ax.add_patch(new_patch)

                if show_id:
                    save_ax.text(
                        pos_real[0], pos_real[1],
                        str(obs_id),
                        color='white',
                        ha='center', va='center',
                        fontsize=13,
                        fontweight='bold',
                        zorder=2
                    )

        if show_targets:
            for robot_id, final_pos in self.final_robot_positions.items():
                if robot_id in self.robot_artists:
                    color = self.robot_artists[robot_id]['color']
                    target_marker = Circle(
                        final_pos,
                        0.1,
                        facecolor='none',
                        edgecolor=color,
                        linewidth=1,
                        linestyle='-',
                        zorder=3
                    )
                    save_ax.add_patch(target_marker)

        for robot_id, robot_data in self.robot_artists.items():
            if use_current_positions:
                pos = robot_data['current_pos']
            else:
                pos = self.initial_robot_states[robot_id]['pos_real']

            radius = robot_data['radius']
            color = robot_data['color']
            show_orientation = robot_data['show_orientation']
            show_id = robot_data['show_id']

            new_body = Circle(
                pos, radius,
                facecolor=color,
                edgecolor='k',
                alpha=0.8,
                zorder=10
            )
            save_ax.add_patch(new_body)

            if show_orientation:
                orientation = 0.0
                arrow_len = radius * 1.5
                dx = arrow_len * np.cos(orientation)
                dy = arrow_len * np.sin(orientation)

                arrow_path = Path([pos, pos + [dx, dy]], [Path.MOVETO, Path.LINETO])
                new_arrow = PathPatch(
                    arrow_path,
                    facecolor='none',
                    edgecolor=color,
                    linewidth=2,
                    zorder=11
                )
                save_ax.add_patch(new_arrow)

            if include_trajectories and robot_data['traj_data'] is not None and len(robot_data['traj_data']) > 1:
                traj_data = np.array(robot_data['traj_data'])
                save_ax.plot(
                    traj_data[:, 0], traj_data[:, 1],
                    color=color,
                    linewidth=1.5,
                    alpha=0.6,
                    zorder=5
                )

            if show_id:
                save_ax.text(
                    pos[0], pos[1] + radius * 1.5,
                    f"ID: {robot_id}",
                    color=color,
                    ha='center', va='bottom',
                    fontsize=13,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', pad=1),
                    zorder=12
                )

        save_ax.set_xlim(self.ax.get_xlim())
        save_ax.set_ylim(self.ax.get_ylim())
        save_ax.set_aspect('equal')
        save_ax.grid(True)

        save_ax.set_xlabel('x (m)', fontsize=13)
        save_ax.set_ylabel('y (m)', fontsize=13)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        save_fig.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close(save_fig)
        print("PDF saved")

    def save_last_frame_as_pdf(self, filename: str = 'last_frame.pdf'):
        self._save_frame_as_pdf(use_current_positions=True, filename=filename,
                                include_trajectories=True, show_targets=False)

    def save_initial_frame_as_pdf(self, filename: str = 'initial_frame.pdf'):
        self._save_frame_as_pdf(use_current_positions=False, filename=filename,
                                include_trajectories=False, show_targets=False)