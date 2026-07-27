"""Stage D: 3D track association across frames.

Stage A already assigns per-frame track IDs via 2D IoU/Hungarian matching,
so most identity-carrying work is done by the time a box reaches here. This
stage handles the two failure modes that carrying the 2D ID blindly would
miss:

- an implausible center jump in one frame (almost always a bad 2D
  association -- e.g. the 2D box hopped onto a neighboring object) gets
  rejected rather than corrupting the trajectory.
- an id-switch after occlusion: Stage A's Tracker2D never reuses an old
  track_id once it's dropped, so a real object that gets occluded for a few
  frames reappears under a brand-new 2D id. If that new id's first 3D center
  is close to where a recently-dropped track was last seen, we relink it to
  the old identity instead of starting a spurious new track.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Track3D:
    track_id: int
    frames: list = field(default_factory=list)
    centers: list = field(default_factory=list)
    dims: list = field(default_factory=list)
    headings: list = field(default_factory=list)
    last_frame: int = -1

    def append(self, frame_idx, center, dims, heading):
        self.frames.append(frame_idx)
        self.centers.append(center)
        self.dims.append(dims)
        self.headings.append(heading)
        self.last_frame = frame_idx


@dataclass
class Tracker3D:
    max_center_jump: float = 4.0  # meters/frame; above this a 2D-track update is rejected as implausible
    reassoc_center_thresh: float = 3.0  # meters; max gap to re-link a dropped track to a new 2D id
    reassoc_max_gap_frames: int = 10  # how long a dropped track stays eligible for re-linking

    _tracks: dict = field(default_factory=dict, init=False)  # our 3D track id -> Track3D
    _id_map: dict = field(default_factory=dict, init=False)  # Stage-A track_id -> our 3D track id
    _next_id: int = field(default=0, init=False)

    def update(self, frame_idx: int, detections):
        """detections: list of (track_id_2d, center(3,), dims(3,), heading)."""
        for track_id_2d, center, dims, heading in detections:
            center = np.asarray(center)

            our_id = self._id_map.get(track_id_2d)
            if our_id is not None and our_id in self._tracks:
                track = self._tracks[our_id]
                gap_frames = max(frame_idx - track.last_frame, 1)
                if np.linalg.norm(center - track.centers[-1]) <= self.max_center_jump * gap_frames:
                    track.append(frame_idx, center, dims, heading)
                else:
                    pass  # implausible jump: drop this update, leave the track waiting
                continue

            relinked = self._try_relink(frame_idx, center)
            if relinked is not None:
                self._id_map[track_id_2d] = relinked
                self._tracks[relinked].append(frame_idx, center, dims, heading)
                continue

            new_id = self._next_id
            self._next_id += 1
            self._id_map[track_id_2d] = new_id
            self._tracks[new_id] = Track3D(track_id=new_id)
            self._tracks[new_id].append(frame_idx, center, dims, heading)

    def _try_relink(self, frame_idx: int, center: np.ndarray):
        best_id, best_dist = None, self.reassoc_center_thresh
        for tid, track in self._tracks.items():
            gap = frame_idx - track.last_frame
            if gap <= 0 or gap > self.reassoc_max_gap_frames:
                continue
            dist = np.linalg.norm(center - track.centers[-1])
            if dist < best_dist:
                best_dist, best_id = dist, tid
        return best_id

    def trajectories(self, min_frames: int = 1) -> dict:
        return {
            tid: {"frames": t.frames, "centers": t.centers, "dims": t.dims, "headings": t.headings}
            for tid, t in self._tracks.items()
            if len(t.frames) >= min_frames
        }
