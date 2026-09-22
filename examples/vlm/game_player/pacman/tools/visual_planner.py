# SPDX-License-Identifier: Apache-2.0

"""Bounded Pacman guidance reconstructed exclusively from rendered RGB pixels."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from maapacman.env.state import Position
    from maapacman.planner import EdwardPlanner, EdwardSafetyRefusal
except ModuleNotFoundError:
    # `cli.py prepare` installs this optional game package before actual play.
    # Keeping imports available here lets the pixel fallback and static tests run.
    @dataclass(frozen=True, order=True)
    class Position:  # type: ignore[no-redef]
        row: int
        col: int

    EdwardPlanner = None  # type: ignore[assignment,misc]

    class EdwardSafetyRefusal(RuntimeError):  # type: ignore[no-redef]
        pass


_TILE_PIXELS = 16
_MAZE_BOTTOM = 384
_DELTAS = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
_OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_NORMAL_GHOST_COLORS = (
    (255, 0, 0),
    (255, 128, 255),
    (128, 255, 255),
    (255, 128, 0),
)


@dataclass(frozen=True)
class _Component:
    area: int
    top: int
    left: int
    bottom: int
    right: int
    points: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class VisualLevel:
    """Minimal Edward-compatible topology recovered from the reset frame."""

    width: int
    height: int
    open_tiles: frozenset[Position]
    ghost_tiles: frozenset[Position]
    pellets: frozenset[Position]
    power_pellets: frozenset[Position]
    pacman_start: Position
    ghost_spawns: tuple[Position, ...]
    horizontal_doors: tuple[Position, Position]
    vertical_doors: tuple[Position, Position]

    def is_wall(self, position: Position, *, actor: str = "pacman") -> bool:
        if actor not in {"pacman", "ghost", "vulnerable", "eyes"}:
            raise ValueError(f"unknown maze actor: {actor!r}")
        allowed = self.open_tiles if actor == "pacman" else self.ghost_tiles
        return position not in allowed

    def portal_exit(self, position: Position, action: str) -> Position:
        if position in self.horizontal_doors:
            other = next(item for item in self.horizontal_doors if item != position)
            if action == "L":
                return Position(other.row, other.col - 1)
            if action == "R":
                return Position(other.row, other.col + 1)
        if position in self.vertical_doors:
            other = next(item for item in self.vertical_doors if item != position)
            if action == "U":
                return Position(other.row - 1, other.col)
            if action == "D":
                return Position(other.row + 1, other.col)
        return position


@dataclass(frozen=True)
class VisualPlan:
    """A concise hint whose complete dynamic input is the screenshot."""

    action: str
    strategy: str
    pacman_cell: tuple[int, int]
    open_actions: tuple[str, ...]
    normal_ghost_cells: tuple[tuple[int, int], ...]
    vulnerable_ghost_cells: tuple[tuple[int, int], ...]
    visible_normal_pellets: int
    visible_power_pellets: int
    planner: str
    option_id: str
    target_cell: tuple[int, int]
    action_sequence: tuple[str, ...]
    commit_moves: int
    safe_actions: tuple[str, ...] = ()

    def prompt_text(self) -> str:
        open_text = "".join(self.open_actions)
        threat_text = (
            ",".join(f"{row}:{col}" for row, col in self.normal_ghost_cells) or "none"
        )
        route_text = ",".join(self.action_sequence)
        return (
            f"Pixel plan: P={self.pacman_cell[0]},{self.pacman_cell[1]} "
            f"open={open_text} ghosts={threat_text} recommend={self.action} "
            f"mode={self.strategy[:1]}. Visual OPTION {self.option_id}: "
            f"target={self.target_cell[0]},{self.target_cell[1]} "
            f"route={route_text} commit={self.commit_moves} "
            f"safe={''.join(self.safe_actions)}."
        )

    def as_evidence(self) -> dict[str, Any]:
        return {
            "source": "rgb_pixels_only",
            "planner": self.planner,
            "recommended_action": self.action,
            "strategy": self.strategy,
            "pacman_cell_row_col": list(self.pacman_cell),
            "open_actions": list(self.open_actions),
            "normal_ghost_cells_row_col": [
                list(cell) for cell in self.normal_ghost_cells
            ],
            "vulnerable_ghost_cells_row_col": [
                list(cell) for cell in self.vulnerable_ghost_cells
            ],
            "visible_normal_pellets": self.visible_normal_pellets,
            "visible_power_pellets": self.visible_power_pellets,
            "option_id": self.option_id,
            "target_cell_row_col": list(self.target_cell),
            "action_sequence": list(self.action_sequence),
            "commit_moves": self.commit_moves,
            "safe_actions": list(self.safe_actions),
        }


def detect_ghost_mode_signature(image: Any) -> tuple[int, int] | None:
    """Return RGB-only lethal/vulnerable ghost counts, or fail on bad pixels."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    normal = sum(_actor_cell(rgb, color) is not None for color in _NORMAL_GHOST_COLORS)
    vulnerable = sum(
        _actor_cell(rgb, color, minimum_area=minimum_area) is not None
        for color, minimum_area in (((50, 50, 255), 20), ((255, 255, 255), 40))
    )
    return normal, vulnerable


def _components(mask: np.ndarray) -> list[_Component]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    result: list[_Component] = []
    for start_y, start_x in zip(*np.nonzero(mask), strict=True):
        if visited[start_y, start_x]:
            continue
        queue = [(int(start_y), int(start_x))]
        visited[start_y, start_x] = True
        points: list[tuple[int, int]] = []
        for y, x in queue:
            points.append((y, x))
            for next_y, next_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if (
                    0 <= next_y < height
                    and 0 <= next_x < width
                    and mask[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_y, next_x))
        ys = [point[0] for point in points]
        xs = [point[1] for point in points]
        result.append(
            _Component(
                area=len(points),
                top=min(ys),
                left=min(xs),
                bottom=max(ys),
                right=max(xs),
                points=tuple(points),
            )
        )
    return sorted(result, key=lambda item: item.area, reverse=True)


def _exact_color(image: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    return np.all(image[..., :3] == color, axis=2)


def detect_pacman_cell(image: Any) -> tuple[int, int] | None:
    """Return one unambiguous maze Pacman cell while excluding dots and HUD lives."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    red, green, blue = (rgb[..., index] for index in range(3))
    yellow = (
        (red >= 180)
        & (green >= 160)
        & (blue <= 110)
        & (np.abs(red.astype(np.int16) - green.astype(np.int16)) <= 90)
    )
    maze_bottom = _MAZE_BOTTOM if rgb.shape[0] >= _MAZE_BOTTOM else rgb.shape[0] * 0.94
    candidates = []
    for component in _components(yellow):
        span = max(
            component.right - component.left + 1,
            component.bottom - component.top + 1,
        )
        if (
            component.area >= 24
            and component.bottom < maze_bottom
            and span <= min(rgb.shape[:2]) * 0.16
        ):
            candidates.append(component)
    if not candidates:
        return None
    largest = candidates[0]
    if len(candidates) > 1 and candidates[1].area >= largest.area * 0.7:
        return None
    return (
        round(largest.top / _TILE_PIXELS),
        round(largest.left / _TILE_PIXELS),
    )


def _actor_cell(
    image: np.ndarray,
    color: tuple[int, int, int],
    *,
    minimum_area: int = 20,
) -> Position | None:
    candidates = []
    for component in _components(_exact_color(image, color)):
        height = component.bottom - component.top + 1
        width = component.right - component.left + 1
        if (
            component.area >= minimum_area
            and component.top < min(_MAZE_BOTTOM, image.shape[0])
            and component.bottom < min(_MAZE_BOTTOM, image.shape[0])
            and height >= 5
            and width >= 5
            and height <= 20
            and width <= 20
        ):
            candidates.append(component)
    if len(candidates) != 1:
        return None
    component = candidates[0]
    return Position(
        round(component.top / _TILE_PIXELS),
        round(component.left / _TILE_PIXELS),
    )


def _pellets(image: np.ndarray) -> tuple[frozenset[Position], frozenset[Position]]:
    normal: set[Position] = set()
    power: set[Position] = set()
    for component in _components(_exact_color(image, (255, 255, 180))):
        if component.bottom >= min(_MAZE_BOTTOM, image.shape[0]):
            continue
        mean_y = sum(point[0] for point in component.points) / component.area
        mean_x = sum(point[1] for point in component.points) / component.area
        position = Position(
            round((mean_y - 7.5) / _TILE_PIXELS),
            round((mean_x - 7.5) / _TILE_PIXELS),
        )
        (normal if component.area <= 8 else power).add(position)
    return frozenset(normal), frozenset(power)


def _ghost_gate(image: np.ndarray) -> Position | None:
    red = _exact_color(image, (255, 0, 0))
    runs: list[tuple[int, int, int]] = []
    maze_height = min(_MAZE_BOTTOM, image.shape[0])
    for y in range(maze_height):
        xs = np.flatnonzero(red[y])
        groups = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1) if len(xs) else []
        for group in groups:
            if len(group) >= 15:
                runs.append((y, int(group[0]), int(group[-1])))
    pairs = [
        (first, second)
        for first in runs
        for second in runs
        if second[0] - first[0] == 3 and first[1:] == second[1:]
    ]
    if not pairs:
        return None
    first, second = max(pairs, key=lambda pair: pair[0][0])
    return Position(
        round(((first[0] + second[0]) / 2 - 8) / _TILE_PIXELS),
        round(((first[1] + first[2]) / 2 - 7.5) / _TILE_PIXELS),
    )


def reconstruct_visual_level(image: Any) -> VisualLevel | None:
    """Recover the fixed grid and walkable graph, or fail closed."""
    rgb = np.asarray(image)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        return None
    rows = rgb.shape[0] // _TILE_PIXELS
    cols = rgb.shape[1] // _TILE_PIXELS
    pacman = detect_pacman_cell(rgb)
    gate = _ghost_gate(rgb)
    if rows < 3 or cols < 3 or pacman is None or gate is None:
        return None
    start = Position(*pacman)
    wall = _exact_color(rgb, (0, 200, 200)) | _exact_color(rgb, (0, 120, 120))

    def edge_open(source: Position, target: Position) -> bool:
        if source == gate or target == gate:
            return False
        y, x = source.row * _TILE_PIXELS + 8, source.col * _TILE_PIXELS + 8
        yy, xx = target.row * _TILE_PIXELS + 8, target.col * _TILE_PIXELS + 8
        return not wall[
            min(y, yy) : max(y, yy) + 1,
            min(x, xx) : max(x, xx) + 1,
        ].any()

    queue = deque([start])
    walkable = {start}
    while queue:
        source = queue.popleft()
        for row_delta, col_delta in _DELTAS.values():
            target = Position(source.row + row_delta, source.col + col_delta)
            if (
                0 <= target.row < rows
                and 0 <= target.col < cols
                and target not in walkable
                and edge_open(source, target)
            ):
                walkable.add(target)
                queue.append(target)
    normal_pellets, power_pellets = _pellets(rgb)
    ghost_spawns = tuple(
        position
        for color in _NORMAL_GHOST_COLORS
        if (position := _actor_cell(rgb, color)) is not None
    )
    if not walkable or not normal_pellets or not ghost_spawns:
        return None
    ghost_tiles = frozenset(walkable | set(ghost_spawns) | {gate})
    return VisualLevel(
        width=cols,
        height=rows,
        open_tiles=frozenset(walkable),
        ghost_tiles=ghost_tiles,
        pellets=normal_pellets,
        power_pellets=power_pellets,
        pacman_start=start,
        ghost_spawns=ghost_spawns,
        horizontal_doors=(Position(rows // 2, 0), Position(rows // 2, cols - 1)),
        vertical_doors=(Position(0, cols // 2), Position(rows - 1, cols // 2)),
    )


class VisualActionSpace:
    """Cache pixel-derived topology for action validation and frame tracking."""

    def __init__(self) -> None:
        self.level: VisualLevel | None = None

    def _ensure_level(self, image: Any) -> VisualLevel:
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError("visual action extraction requires an RGB image")
        if self.level is None:
            self.level = reconstruct_visual_level(rgb)
        if self.level is None:
            raise ValueError("could not reconstruct Pacman topology from RGB pixels")
        return self.level

    def available_actions(self, image: Any) -> tuple[str, ...]:
        """Extract legal Pacman actions from RGB pixels, failing on ambiguity."""
        level = self._ensure_level(image)
        pacman_cell = detect_pacman_cell(image)
        if pacman_cell is None:
            raise ValueError("could not locate Pacman unambiguously from RGB pixels")
        player = Position(*pacman_cell)
        if level.is_wall(player, actor="pacman"):
            raise ValueError("pixel-derived Pacman cell is outside the walkable graph")
        return tuple(
            action
            for action, (row_delta, col_delta) in _DELTAS.items()
            if not level.is_wall(
                Position(player.row + row_delta, player.col + col_delta),
                actor="pacman",
            )
        )

    def is_expected_transition(
        self,
        previous_cell: tuple[int, int],
        current_cell: tuple[int, int],
        action: str,
    ) -> bool:
        """Recognize an adjacent or portal move; an unchanged cell is a wall no-op."""
        if current_cell == previous_cell:
            return True
        if self.level is None or action not in _DELTAS:
            return False
        previous = Position(*previous_cell)
        row_delta, col_delta = _DELTAS[action]
        candidate = Position(previous.row + row_delta, previous.col + col_delta)
        if self.level.is_wall(candidate, actor="pacman"):
            return False
        return self.level.portal_exit(candidate, action) == Position(*current_cell)


class StandaloneVisualPlanner:
    """Stateful, screenshot-only action adviser for the standalone player."""

    def __init__(self) -> None:
        self.level: VisualLevel | None = None
        self.planner: Any | None = None
        self.previous_action: str | None = None
        self._previous_normal_ghosts: dict[int, Position] = {}

    def record_model_action(self, action: str) -> None:
        """Track the action the model actually chose, not the advisory action."""
        if action not in _DELTAS:
            raise ValueError(f"invalid visual planner action: {action!r}")
        self.previous_action = action
        if self.planner is not None:
            self.planner.record_action(action)

    def _open_actions(self, player: Position) -> tuple[str, ...]:
        assert self.level is not None
        return tuple(
            action
            for action, (row_delta, col_delta) in _DELTAS.items()
            if not self.level.is_wall(
                Position(player.row + row_delta, player.col + col_delta),
                actor="pacman",
            )
        )

    def _ghosts(
        self, image: np.ndarray, player: Position
    ) -> tuple[
        list[dict[str, Any]], tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]
    ]:
        assert self.level is not None
        ghosts: list[dict[str, Any]] = []
        normal_cells: list[tuple[int, int]] = []
        current_normal: dict[int, Position] = {}
        for entity_id, color in enumerate(_NORMAL_GHOST_COLORS):
            position = _actor_cell(image, color)
            if position is None:
                continue
            current_normal[entity_id] = position
            normal_cells.append((position.row, position.col))
            ghosts.append(
                {
                    "id": entity_id,
                    "position": [position.row, position.col],
                    "state": "normal",
                }
            )
            previous = self._previous_normal_ghosts.get(entity_id)
            if previous is None or previous == position:
                continue
            distance_to_player = self._distance(position, player, actor="ghost")
            row_delta = position.row - previous.row
            col_delta = position.col - previous.col
            if (
                distance_to_player is None
                or distance_to_player > 6
                or abs(row_delta) + abs(col_delta) != 1
            ):
                continue
            forward = Position(position.row + row_delta, position.col + col_delta)
            if not self.level.is_wall(forward, actor="ghost"):
                ghosts.append(
                    {
                        "id": f"{entity_id}-visual-forward",
                        "position": [forward.row, forward.col],
                        "state": "normal",
                    }
                )
        self._previous_normal_ghosts = current_normal

        vulnerable_positions = []
        for color, minimum_area in (((50, 50, 255), 20), ((255, 255, 255), 40)):
            position = _actor_cell(image, color, minimum_area=minimum_area)
            if position is not None:
                vulnerable_positions.append(position)
                ghosts.append(
                    {
                        "id": f"vulnerable-{len(vulnerable_positions) - 1}",
                        "position": [position.row, position.col],
                        "state": "vulnerable",
                    }
                )
        vulnerable_cells = tuple(
            (position.row, position.col) for position in vulnerable_positions
        )
        return ghosts, tuple(normal_cells), vulnerable_cells

    def _distance(
        self, source: Position, target: Position, *, actor: str = "pacman"
    ) -> int | None:
        assert self.level is not None
        queue = deque([(source, 0)])
        visited = {source}
        while queue:
            current, distance = queue.popleft()
            if current == target:
                return distance
            for row_delta, col_delta in _DELTAS.values():
                candidate = Position(
                    current.row + row_delta,
                    current.col + col_delta,
                )
                if candidate in visited or self.level.is_wall(candidate, actor=actor):
                    continue
                visited.add(candidate)
                queue.append((candidate, distance + 1))
        return None

    def _fallback_action(
        self,
        player: Position,
        open_actions: tuple[str, ...],
        normal_ghosts: tuple[tuple[int, int], ...],
        visible_pellets: frozenset[Position],
    ) -> str | None:
        assert self.level is not None
        if not open_actions:
            return None
        ghost_positions = tuple(Position(*cell) for cell in normal_ghosts)
        pellets = visible_pellets or self.level.pellets
        scored: list[tuple[int, int, int, int, str]] = []
        for action in open_actions:
            row_delta, col_delta = _DELTAS[action]
            target = self.level.portal_exit(
                Position(player.row + row_delta, player.col + col_delta),
                action,
            )
            ghost_distances = [
                distance
                for ghost in ghost_positions
                if (distance := self._distance(ghost, target, actor="ghost"))
                is not None
            ]
            clearance = min(ghost_distances, default=10**6)
            pellet_distances = [
                distance
                for pellet in pellets
                if (distance := self._distance(target, pellet)) is not None
            ]
            pellet_distance = min(pellet_distances, default=None)
            distance_score = -(
                pellet_distance if pellet_distance is not None else 10**6
            )
            degree = sum(
                not self.level.is_wall(
                    Position(target.row + delta_row, target.col + delta_col),
                    actor="pacman",
                )
                for delta_row, delta_col in _DELTAS.values()
            )
            reversal = int(action == _OPPOSITE.get(self.previous_action))
            scored.append((clearance, distance_score, degree, -reversal, action))
        return max(scored)[-1]

    def _shortest_route(
        self, source: Position, target: Position
    ) -> tuple[str, ...] | None:
        """Recover one deterministic route from the RGB-derived topology."""
        assert self.level is not None
        queue = deque([(source, ())])
        visited = {source}
        while queue:
            current, route = queue.popleft()
            if current == target:
                return route
            for action, (row_delta, col_delta) in _DELTAS.items():
                adjacent = Position(
                    current.row + row_delta,
                    current.col + col_delta,
                )
                if self.level.is_wall(adjacent, actor="pacman"):
                    continue
                candidate = self.level.portal_exit(adjacent, action)
                if candidate in visited:
                    continue
                visited.add(candidate)
                queue.append((candidate, (*route, action)))
        return None

    def plan(
        self, image: Any, *, blocked_action: str | None = None
    ) -> VisualPlan | None:
        """Return guidance only when all required screenshot extraction is clear."""
        rgb = np.asarray(image)
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            return None
        if self.level is None:
            self.level = reconstruct_visual_level(rgb)
            if self.level is None:
                return None
            self.planner = (
                EdwardPlanner(self.level) if EdwardPlanner is not None else None
            )
        pacman_cell = detect_pacman_cell(rgb)
        if pacman_cell is None:
            return None
        player = Position(*pacman_cell)
        if self.level.is_wall(player, actor="pacman"):
            return None
        open_actions = self._open_actions(player)
        if blocked_action is not None:
            open_actions = tuple(
                action for action in open_actions if action != blocked_action
            )
        if not open_actions:
            return None
        ghosts, normal_ghosts, vulnerable_ghosts = self._ghosts(rgb, player)
        normal_pellets, power_pellets = _pellets(rgb)
        state = {
            "row": player.row,
            "col": player.col,
            "open": open_actions,
            "ghosts": ghosts,
            # Pixel colors establish vulnerability, but not a reliable remaining
            # timer. Keep zero so Edward will not plan a time-sensitive chase.
            "edible_ticks": 0,
        }
        planner_name = "edward_visual"
        target = player
        commit_moves = 1
        route: tuple[str, ...] = ()
        safe_actions: tuple[str, ...] = ()
        try:
            if self.planner is None:
                raise EdwardSafetyRefusal("Edward planner is unavailable")
            decision = self.planner.decide(state)
            action = decision.action
            selected = next(
                candidate
                for candidate in decision.candidates
                if candidate.option_id == decision.option_id
            )
            safe_actions = tuple(
                action
                for action in open_actions
                if any(
                    candidate.first_action == action
                    for candidate in decision.candidates
                )
            )
            strategy = selected.strategy
            target = Position(*selected.target)
            candidate_route = self._shortest_route(player, target)
            if not candidate_route or candidate_route[0] != action:
                raise ValueError("visual route does not match Edward's first action")
            commit_moves = min(8, selected.commit_moves, len(candidate_route))
            route = candidate_route[:commit_moves]
        except (EdwardSafetyRefusal, ValueError, StopIteration):
            planner_name = "visual_safety_fallback"
            action = self._fallback_action(
                player,
                open_actions,
                normal_ghosts,
                normal_pellets,
            )
            strategy = "AVOID" if normal_ghosts else "COLLECT"
            safe_actions = (action,) if action is not None else ()
        if action not in open_actions:
            return None
        if not route:
            row_delta, col_delta = _DELTAS[action]
            target = self.level.portal_exit(
                Position(player.row + row_delta, player.col + col_delta),
                action,
            )
            route = (action,)
            commit_moves = 1
        if not safe_actions:
            safe_actions = (action,)
        return VisualPlan(
            action=action,
            strategy=strategy,
            pacman_cell=pacman_cell,
            open_actions=open_actions,
            normal_ghost_cells=normal_ghosts,
            vulnerable_ghost_cells=vulnerable_ghosts,
            visible_normal_pellets=len(normal_pellets),
            visible_power_pellets=len(power_pellets),
            planner=planner_name,
            option_id="A0",
            target_cell=(target.row, target.col),
            action_sequence=route,
            commit_moves=commit_moves,
            safe_actions=safe_actions,
        )
