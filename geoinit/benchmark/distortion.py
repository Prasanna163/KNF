"""Randomised geometry distortion generator for GeoInit benchmarking.

Provides functions to introduce realistic bond, angle, and fragment-level
noise to test structures.
"""

from __future__ import annotations

import numpy as np

from geoinit.core.topology import Topology


def generate_distorted_coords(
    symbols: list[str],
    coords: np.ndarray,
    topology: Topology,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a realistically distorted geometry from a reference starting structure.

    Parameters
    ----------
    symbols : list[str]
        Atomic element symbols.
    coords : np.ndarray, shape (N, 3)
        Reference atomic coordinates in Å.
    topology : Topology
        Molecular topology.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray, shape (N, 3)
        Distorted atomic coordinates in Å.
    """
    coords = np.asarray(coords, dtype=np.float64)
    n_atoms = len(symbols)
    distorted = coords.copy()

    # 1. Add internal Cartesian noise to all atoms
    # A standard deviation between 0.04 and 0.08 Å corresponds to standard
    # bond distortions of ~0.05 to ~0.15 Å.
    std = rng.uniform(0.04, 0.08)
    distorted += rng.normal(0, std, size=coords.shape)

    # 2. Fragment translation and rotation noise (only for complexes)
    from collections import defaultdict
    adj = defaultdict(list)
    for b in topology.bonds:
        adj[b.i].append(b.j)
        adj[b.j].append(b.i)

    visited = set()
    fragments = []
    for i in range(n_atoms):
        if i not in visited:
            comp = []
            queue = [i]
            visited.add(i)
            while queue:
                curr = queue.pop(0)
                comp.append(curr)
                for nbr in adj[curr]:
                    if nbr not in visited:
                        visited.add(nbr)
                        queue.append(nbr)
            fragments.append(sorted(comp))

    if len(fragments) >= 2:
        # Keep fragment 0 (host) anchored.
        # Apply random translations and rotations to all guest fragments.
        for f_idx in range(1, len(fragments)):
            frag = fragments[f_idx]
            com = np.mean(distorted[frag], axis=0)

            # Translate guest fragment: translation distance between 0.5 and 1.5 Å
            t_len = rng.uniform(0.5, 1.5)
            t_dir = rng.normal(size=3)
            norm = np.linalg.norm(t_dir)
            if norm > 0.0:
                t_dir /= norm
            else:
                t_dir = np.array([1.0, 0.0, 0.0])
            distorted[frag] += t_len * t_dir

            # Rotate guest fragment around its COM: rotation angle between 10° and 45°
            angle = rng.uniform(np.radians(10), np.radians(45))
            rot_axis = rng.normal(size=3)
            norm_rot = np.linalg.norm(rot_axis)
            if norm_rot > 0.0:
                rot_axis /= norm_rot
            else:
                rot_axis = np.array([0.0, 0.0, 1.0])

            # Rodrigues' rotation formula
            K = np.array([
                [0, -rot_axis[2], rot_axis[1]],
                [rot_axis[2], 0, -rot_axis[0]],
                [-rot_axis[1], rot_axis[0], 0]
            ])
            R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)

            distorted[frag] = (distorted[frag] - com) @ R.T + com

    return distorted
