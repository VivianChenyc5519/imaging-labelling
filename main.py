import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi
from skimage import measure, morphology
import pyvista as pv
import networkx as nx
import SimpleITK as sitk
import copy
from collections import Counter
from bezier_simvc import (
    angle_deg,
    bezier_cubic,
    optimize_bezier_simvc_nd,
    sample_bezier_bridge,
    second_derivative_smoothness,
)

from post_process import merge_ballmerge_voronoi_node_clusters


DISTANCE_WEIGHT = 0.3
XY_DISTANCE_WEIGHT= DISTANCE_WEIGHT * 0.8
Z_DISTANCE_WEIGHT = DISTANCE_WEIGHT * 0.2
CURVATURE_WEIGHT = 0.4
ANGLE_WEIGHT = 0.3

ORIGINAL_EDGES = 0
REPAIR_EDGES = 1
SEPARATE_COMPONENTS = 2
PROBLEMATIC_EDGES = 3

# ============================================================
# Load CT
# ============================================================

def load_nii(path):
    img = nib.load(path)
    ct = img.get_fdata().astype(np.float32)
    affine = img.affine
    spacing = img.header.get_zooms()[:3]
    return ct, affine, spacing


# ============================================================
# Airway / bronchi extraction
# ============================================================

def find_trachea_seed(ct, air_thr=-850, top_frac=0.25):
    sx, sy, sz = ct.shape
    z0 = int(sz * (1.0 - top_frac))

    air = ct[:, :, z0:] < air_thr
    labels = measure.label(air, connectivity=3)
    props = measure.regionprops(labels)

    center = np.array([sx / 2, sy / 2, air.shape[2] / 2])

    best = None
    best_score = np.inf

    for p in props:
        if p.area < 50 or p.area > 100000:
            continue

        c = np.array(p.centroid)
        dist = np.linalg.norm((c - center) / np.array([sx, sy, air.shape[2]]))
        score = dist - 1e-5 * p.area

        if score < best_score:
            best_score = score
            best = p

    if best is None:
        raise RuntimeError("Could not find trachea seed. Use --seed x y z.")

    c = np.round(best.centroid).astype(int)

    return int(c[0]), int(c[1]), int(c[2] + z0)


def extract_airway_from_segmentation(seg, airway_labels=None, min_voxels=0):
    seg = np.nan_to_num(seg)

    if airway_labels is None:
        airway = seg > 0
    else:
        airway = np.isin(seg, airway_labels)

    if airway.sum() == 0:
        raise RuntimeError("Airway segmentation mask is empty.")

    if min_voxels > 0:
        airway = morphology.remove_small_objects(
            airway.astype(bool),
            min_size=min_voxels,
            connectivity=3,
        )

    return airway.astype(bool)

def extract_airway_only(
    ct,
    seed,
    spacing,
    air_low=-1024,
    air_high=-950,
    max_radius_mm=12.0,
    min_voxels=800,
):
    candidate_air = (ct >= -1024) & (ct <= -950)

    if not candidate_air[seed]:
        raise RuntimeError(
            f"Seed {seed} is not inside air candidate. "
            f"Seed HU={ct[seed]}. Try --air-high -850 or another seed."
        )

    print("Computing distance transform...")
    radius_mm = ndi.distance_transform_edt(
        candidate_air,
        sampling=spacing
    )

    # Keeps tube-like structures, removes large lung air cavities
    tube_like_air = candidate_air & (radius_mm <= max_radius_mm)

    if not tube_like_air[seed]:
        raise RuntimeError(
            f"Seed radius is {radius_mm[seed]:.2f} mm, larger than "
            f"max_radius_mm={max_radius_mm}. Increase --max-radius-mm."
        )

    print("Extracting component connected to trachea...")
    labels, _ = ndi.label(
        tube_like_air,
        structure=np.ones((3, 3, 3), dtype=np.uint8)
    )

    seed_label = labels[seed]

    if seed_label == 0:
        raise RuntimeError("Seed component not found after filtering.")

    airway = labels == seed_label

    airway = morphology.remove_small_objects(
        airway.astype(bool),
        min_size=5000,
        connectivity=3
    )

    airway = ndi.binary_closing(
        airway,
        structure=np.ones((3,3,3)),
        iterations=1
    )

    print("Final airway voxels:", int(airway.sum()))

    if airway.sum() == 0:
        raise RuntimeError("Airway mask is empty.")

    return airway.astype(bool)


# ============================================================
# Skeletonization
# ============================================================

def skeletonize_mask(mask):
    if hasattr(morphology, "skeletonize_3d"):
        return morphology.skeletonize_3d(mask).astype(bool)

    return morphology.skeletonize(mask, method="lee").astype(bool)


# ============================================================
# Graph construction
# ============================================================

NEIGHBOR_OFFSETS_26 = [
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if not (dx == 0 and dy == 0 and dz == 0)
]


def in_bounds(p, shape):
    i, j, k = p
    return (
        0 <= i < shape[0] and
        0 <= j < shape[1] and
        0 <= k < shape[2]
    )


def remove_loops_keep_tree(G, root_node):
    """
    Convert G into a tree rooted at root_node by removing cycle-closing edges.

    Keeps the shortest-path tree from root_node.
    This is safer than merging cycle nodes because it preserves existing
    branch geometry and guarantees no graph cycles.
    """
    if root_node not in G:
        raise RuntimeError("root_node is not in G.")

    H = G.copy()

    # Edge weight = physical edge length.
    for u, v, data in H.edges(data=True):
        pts = np.asarray(data.get("points_xyz", []), dtype=np.float32)

        if len(pts) >= 2:
            length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
        else:
            length = float(
                np.linalg.norm(
                    np.asarray(H.nodes[u]["xyz"]) -
                    np.asarray(H.nodes[v]["xyz"])
                )
            )

        data["tree_weight"] = max(length, 1e-6)

    # Minimum spanning tree removes loops.
    T = nx.minimum_spanning_tree(H, weight="tree_weight")

    removed_edges = [
        (u, v)
        for u, v in H.edges()
        if not T.has_edge(u, v)
    ]

    print(f"Removed {len(removed_edges)} loop-closing edges.")

    return T, root_node

def edge_polyline_length(data, G=None, u=None, v=None):
    pts = np.asarray(data.get("points_xyz", []), dtype=np.float64)

    if len(pts) >= 2:
        return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))

    if G is not None and u is not None and v is not None:
        return float(
            np.linalg.norm(
                np.asarray(G.nodes[u]["xyz"], dtype=np.float64)
                - np.asarray(G.nodes[v]["xyz"], dtype=np.float64)
            )
        )

    return 0.0


def replace_polyline_endpoint(points, old_xyz, new_xyz):
    """
    Replace whichever endpoint of the polyline is closer to old_xyz
    with new_xyz. This keeps the external edge geometry mostly intact
    after merging loop nodes.
    """
    pts = np.asarray(points, dtype=np.float64)

    if len(pts) == 0:
        return pts

    old_xyz = np.asarray(old_xyz, dtype=np.float64)
    new_xyz = np.asarray(new_xyz, dtype=np.float64)

    d0 = np.linalg.norm(pts[0] - old_xyz)
    d1 = np.linalg.norm(pts[-1] - old_xyz)

    pts = pts.copy()

    if d0 <= d1:
        pts[0] = new_xyz
    else:
        pts[-1] = new_xyz

    return pts.astype(np.float32)


def get_skeleton_neighbors(p, skeleton):
    out = []

    for d in NEIGHBOR_OFFSETS_26:
        q = (
            p[0] + d[0],
            p[1] + d[1],
            p[2] + d[2],
        )

        if in_bounds(q, skeleton.shape) and skeleton[q]:
            out.append(q)

    return out


def ijk_to_xyz(p, spacing):
    return np.array(
        [
            p[0] * spacing[0],
            p[1] * spacing[1],
            p[2] * spacing[2],
        ],
        dtype=np.float32
    )

def choose_root_node(G, seed=None):
    if G.number_of_nodes() == 0:
        raise RuntimeError("Cannot choose root from empty graph.")

    # Best option for both airway and vessel if a reliable seed exists
    if seed is not None:
        seed = np.asarray(seed, dtype=np.float32)
        return min(
            G.nodes,
            key=lambda n: np.linalg.norm(
                G.nodes[n]["ijk"].astype(np.float32) - seed
            )
        )

    # Airway fallback: top of trachea / most superior node
    return max(
        G.nodes,
        key=lambda n: G.nodes[n]["ijk"][2]
    )

def skeleton_to_graph(skeleton, spacing, radius_map_mm=None):
    """
    Nodes:
        skeleton voxels with degree != 2
        endpoint degree = 1
        bifurcation degree >= 3

    Edges:
        paths of degree-2 voxels between nodes
    """

    skeleton = skeleton.astype(bool)
    skel_points = [tuple(p) for p in np.argwhere(skeleton)]

    degree = {}
    node_voxels = set()
    for p in skel_points:
        nbs = get_skeleton_neighbors(p, skeleton)
        degree[p] = len(nbs)

        if len(nbs) != 2:
            node_voxels.add(p)
    G = nx.Graph()
    voxel_to_node = {}

    node_id = 0

    for p in node_voxels:
        r = 1.5
        if radius_map_mm is not None:
            r = float(radius_map_mm[p])

        G.add_node(
            node_id,
            ijk=np.array(p, dtype=np.int32),
            xyz=ijk_to_xyz(p, spacing),
            radius_mm=r,
            degree=degree[p],
        )

        voxel_to_node[p] = node_id
        node_id += 1
    visited_edges = set()

    for start_voxel in node_voxels:
        start_node = voxel_to_node[start_voxel]

        for nb in get_skeleton_neighbors(start_voxel, skeleton):
            edge_key = frozenset([start_voxel, nb])

            if edge_key in visited_edges:
                continue

            path = [start_voxel, nb]
            visited_edges.add(edge_key)

            prev = start_voxel
            curr = nb

            while curr not in node_voxels:
                nbs = get_skeleton_neighbors(curr, skeleton)

                candidates = [x for x in nbs if x != prev]

                if len(candidates) == 0:
                    break

                nxt = candidates[0]

                visited_edges.add(frozenset([curr, nxt]))

                path.append(nxt)
                prev, curr = curr, nxt

            if curr not in node_voxels:
                continue

            end_node = voxel_to_node[curr]

            if start_node == end_node:
                continue

            path_ijk = np.array(path, dtype=np.int32)
            path_xyz = np.array(
                [ijk_to_xyz(tuple(p), spacing) for p in path_ijk],
                dtype=np.float32
            )

            if radius_map_mm is not None:
                radii = np.array(
                    [radius_map_mm[tuple(p)] for p in path_ijk],
                    dtype=np.float32
                )
            else:
                radii = np.ones(len(path_ijk), dtype=np.float32) * 1.5

            G.add_edge(
                start_node,
                end_node,
                points_ijk=path_ijk,
                points_xyz=path_xyz,
                radius_mm=radii,
                active=True
            )

    # Pick root node as the most superior/high-z node
    root_node = choose_root_node(G)
    return G, root_node



# ============================================================
# Interactive graph repair
# ============================================================

def unit_vector(v):
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < 1e-8:
        return None
    return v / n


def orient_tangent_toward(tangent, desired_direction):
    """
    Flip tangent if needed so it points in the same direction as desired_direction.
    """
    t = unit_vector(tangent)
    d = unit_vector(desired_direction)

    if t is None or d is None:
        return None

    if np.dot(t, d) < 0:
        t = -t

    return t

def angle_deg(v1, v2):
    u1 = unit_vector(v1)
    u2 = unit_vector(v2)
    if u1 is None or u2 is None:
        return 0.0
    c = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def point_to_polyline_distance(point, polyline):
    """Return minimum distance from a 3D point to a polyline."""
    p = np.asarray(point, dtype=np.float64)
    pts = np.asarray(polyline, dtype=np.float64)
    if len(pts) == 0:
        return np.inf
    if len(pts) == 1:
        return float(np.linalg.norm(p - pts[0]))

    best = np.inf
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom < 1e-12:
            d = np.linalg.norm(p - a)
        else:
            t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
            q = a + t * ab
            d = np.linalg.norm(p - q)
        best = min(best, d)
    return float(best)


def unit(v):
    n = np.linalg.norm(v)
    if n < 1e-8:
        return np.zeros_like(v)
    return v / n


def estimate_tangent_from_largest_radius_branch(
    G,
    node,
    exclude_node=None,
    use="mean",   # "mean", "max", or "near_node"
):
    """
    Estimate local tangent at `node` using the adjacent branch with largest radius.

    Returned tangent points away from `node` along that selected branch.
    """
    p0 = np.asarray(G.nodes[node]["xyz"], dtype=np.float32)

    best_nb = None
    best_pts = None
    best_radius = -np.inf

    for nb in G.neighbors(node):
        if exclude_node is not None and nb == exclude_node:
            continue

        data = G.edges[node, nb]

        if not data.get("active", True):
            continue

        pts = np.asarray(data.get("points_xyz", []), dtype=np.float32)
        radii = np.asarray(data.get("radius_mm", []), dtype=np.float32)

        if len(radii) > 0:
            if use == "max":
                radius_score = float(np.nanmax(radii))
            elif use == "near_node":
                # use radius closest to `node`
                if len(pts) >= 2:
                    if np.linalg.norm(pts[0] - p0) <= np.linalg.norm(pts[-1] - p0):
                        radius_score = float(radii[0])
                    else:
                        radius_score = float(radii[-1])
                else:
                    radius_score = float(np.nanmean(radii))
            else:
                radius_score = float(np.nanmean(radii))
        else:
            radius_score = float(G.nodes[nb].get("radius_mm", 0.0))

        if radius_score > best_radius:
            best_radius = radius_score
            best_nb = nb
            best_pts = pts

    if best_nb is None:
        print("Node has no neighbor !")
        return np.zeros(3, dtype=np.float32)

    if best_pts is not None and len(best_pts) >= 2:
        pts = best_pts

        # orient polyline so pts[0] is closest to `node`
        if np.linalg.norm(pts[0] - p0) > np.linalg.norm(pts[-1] - p0):
            pts = pts[::-1]

        return unit(pts[1] - pts[0]).astype(np.float32)

    k = min(5, len(pts) - 1)
    return unit(pts[k] - pts[0]).astype(np.float32)

# def estimate_tangent_from_graph(G, node, exclude_node=None):
#     """
#     Estimate local tangent at a graph node.

#     Tangent points away from `node` along the selected existing branch.
#     Uses the longest adjacent edge after excluding `exclude_node`.
#     """
#     p0 = np.asarray(G.nodes[node]["xyz"], dtype=np.float32)

#     best_nb = None
#     best_pts = None
#     best_len = -1.0

#     for nb in G.neighbors(node):
#         if exclude_node is not None and nb == exclude_node:
#             continue

#         data = G.edges[node, nb]

#         if not data.get("active", True):
#             continue

#         pts = np.asarray(data.get("points_xyz", []), dtype=np.float32)

#         if len(pts) >= 2:
#             length = float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))
#         else:
#             p1 = np.asarray(G.nodes[nb]["xyz"], dtype=np.float32)
#             length = float(np.linalg.norm(p1 - p0))

#         if length > best_len:
#             best_len = length
#             best_nb = nb
#             best_pts = pts

#     if best_nb is None:
#         return np.zeros(3, dtype=np.float32)

#     if best_pts is not None and len(best_pts) >= 2:
#         pts = best_pts

#         # orient polyline so pts[0] is closest to node
#         if np.linalg.norm(pts[0] - p0) > np.linalg.norm(pts[-1] - p0):
#             pts = pts[::-1]

#         return unit(pts[1] - pts[0]).astype(np.float32)

#     p1 = np.asarray(G.nodes[best_nb]["xyz"], dtype=np.float32)
#     return unit(p1 - p0).astype(np.float32)


def nearest_edge_to_point(G, picked_point, active_only=True):
    best = None
    best_dist = np.inf
    for u, v, data in G.edges(data=True):
        if active_only and not data.get("active", True):
            continue
        pts = data.get("points_xyz")
        if pts is None or len(pts) < 2:
            pts = np.vstack([G.nodes[u]["xyz"], G.nodes[v]["xyz"]])
        d = point_to_polyline_distance(picked_point, pts)
        if d < best_dist:
            best_dist = d
            best = (u, v)
    return best, best_dist


def smooth_graph_edge_polylines(G, iterations=5, alpha=0.35):
    for u, v, data in G.edges(data=True):
        pts = np.asarray(data.get("points_xyz", []), dtype=np.float32)

        if len(pts) < 4:
            continue

        smoothed = pts.copy()

        for _ in range(iterations):
            new_pts = smoothed.copy()
            new_pts[1:-1] = (
                (1.0 - alpha) * smoothed[1:-1]
                + alpha * 0.5 * (smoothed[:-2] + smoothed[2:])
            )
            smoothed = new_pts

        data["points_xyz_raw"] = pts
        data["points_xyz"] = smoothed

    return G



def endpoints_in_component(G, comp):
    return [n for n in comp if G.degree[n] == 1]

def endpoint_attached_edge_mean_radius(G, endpoint, default=0.0):
    """
    For a degree-1 endpoint, return the mean radius of its attached edge.

    This is more stable than using only G.nodes[endpoint]["radius_mm"].
    """
    nbs = list(G.neighbors(endpoint))

    if len(nbs) == 0:
        return float(default)

    # Since this is an endpoint, it should normally have exactly one neighbor.
    nb = nbs[0]
    data = G.edges[endpoint, nb]

    r = data.get("radius_mm", None)

    if r is not None:
        arr = np.asarray(r, dtype=np.float64)

        if arr.ndim == 0:
            if not np.isnan(arr):
                return float(arr)

        elif len(arr) > 0 and not np.all(np.isnan(arr)):
            return float(np.nanmean(arr))

    # Fallback if edge radius is missing.
    r0 = float(G.nodes[endpoint].get("radius_mm", default))
    r1 = float(G.nodes[nb].get("radius_mm", default))

    return 0.5 * (r0 + r1)

def largest_radius_endpoints_in_component(G, comp, top_k=3):
    """
    Return degree-1 endpoints in this component, sorted by the mean radius
    of their attached edge.

    This is better than sorting by endpoint node radius alone.
    """
    endpoints = endpoints_in_component(G, comp)

    if len(endpoints) == 0:
        return []

    endpoints = sorted(
        endpoints,
        key=lambda n: endpoint_attached_edge_mean_radius(G, n, default=0.0),
        reverse=True,
    )

    print("Largest-radius source endpoints, using attached-edge mean radius:")
    for n in endpoints[:top_k]:
        r_edge = endpoint_attached_edge_mean_radius(G, n, default=0.0)
        r_node = float(G.nodes[n].get("radius_mm", 0.0))

        print(
            f"  node={n}, "
            f"edge_mean_radius={r_edge:.3f}, "
            f"node_radius={r_node:.3f}, "
            f"degree={G.degree[n]}, "
            f"xyz={G.nodes[n].get('xyz')}"
        )

    return endpoints[:top_k]

def cumulative_lengths(pts):
    pts = np.asarray(pts, dtype=np.float64)
    if len(pts) < 2:
        return np.zeros(len(pts), dtype=np.float64)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def edge_radius_array_for_points(G, node, nb, pts):
    data = G.edges[node, nb]
    radii = data.get("radius_mm", None)

    if radii is None:
        r0 = float(G.nodes[node].get("radius_mm", 0.0))
        r1 = float(G.nodes[nb].get("radius_mm", r0))
        radii = np.linspace(r0, r1, len(pts), dtype=np.float64)
    else:
        radii = np.asarray(radii, dtype=np.float64)

        if radii.ndim == 0:
            radii = np.full(len(pts), float(radii), dtype=np.float64)

        if len(radii) != len(pts):
            r_mean = float(np.nanmean(radii)) if len(radii) > 0 else 0.0
            radii = np.full(len(pts), r_mean, dtype=np.float64)

    return radii


def local_component_tangent(
    G,
    node,
    mode,
    exclude_node=None,
    skip_mm=2.0,
    lookahead_mm=12.0,
    tangent_sample_mm=8.0,
    use_stat="mean",
    return_neighbor=False,
):
    """
    Select attached edge by largest radius after moving away from the junction.

    mode:
        "into_node":
            tangent points from component interior toward node.
            Use for source endpoint.

        "away_from_node":
            tangent points from node into component interior.
            Use for target endpoint.

    skip_mm:
        Ignore radius samples very close to the junction/node.

    lookahead_mm:
        Compare radius samples between skip_mm and lookahead_mm from node.

    tangent_sample_mm:
        Point distance from node used to create tangent vector.
    """

    p_node = np.asarray(G.nodes[node]["xyz"], dtype=np.float64)

    neighbors = [
        nb for nb in G.neighbors(node)
        if nb != exclude_node and G.edges[node, nb].get("active", True)
    ]

    if len(neighbors) == 0:
        return (None, None) if return_neighbor else None

    candidates = []

    for nb in neighbors:
        data = G.edges[node, nb]

        pts = data.get("points_xyz", None)
        if pts is None or len(pts) < 2:
            pts = np.vstack([
                G.nodes[node]["xyz"],
                G.nodes[nb]["xyz"],
            ])

        pts = np.asarray(pts, dtype=np.float64)
        radii = edge_radius_array_for_points(G, node, nb, pts)

        # Orient edge so pts[0] is near current node.
        if np.linalg.norm(pts[0] - p_node) > np.linalg.norm(pts[-1] - p_node):
            pts = pts[::-1]
            radii = radii[::-1]

        s = cumulative_lengths(pts)

        # Radius comparison window away from node.
        mask = (s >= skip_mm) & (s <= lookahead_mm)

        # Fallback if edge is too short.
        if not np.any(mask):
            mask = s <= lookahead_mm

        if not np.any(mask):
            continue

        selected_radii = radii[mask]

        if use_stat == "max":
            radius_score = float(np.nanmax(selected_radii))
        elif use_stat == "median":
            radius_score = float(np.nanmedian(selected_radii))
        else:
            radius_score = float(np.nanmean(selected_radii))

        # Choose tangent sample point by physical distance.
        k = int(np.argmin(np.abs(s - tangent_sample_mm)))
        k = max(1, min(k, len(pts) - 1))

        p0 = pts[0]
        pk = pts[k]

        if mode == "into_node":
            tangent = p0 - pk
        elif mode == "away_from_node":
            tangent = pk - p0
        else:
            raise ValueError("mode must be 'into_node' or 'away_from_node'")

        if np.linalg.norm(tangent) < 1e-8:
            continue

        length = float(s[-1])

        candidates.append({
            "neighbor": nb,
            "tangent": tangent,
            "radius_score": radius_score,
            "length": length,
            "sample_index": k,
            "sample_distance": float(s[k]),
        })

    if len(candidates) == 0:
        return (None, None) if return_neighbor else None

    # Primary priority: largest branch radius.
    # Tie-breaker: longer edge.
    candidates.sort(
        key=lambda c: (c["radius_score"], c["length"]),
        reverse=True,
    )

    best = candidates[0]

    # print(f"\nNode {node}, mode={mode}")
    # for c in candidates:
    #     print(
    #         f"  nb={c['neighbor']} "
    #         f"radius_score={c['radius_score']:.3f} "
    #         f"length={c['length']:.3f} "
    #         f"sample_dist={c['sample_distance']:.3f}"
    #     )
    # print("  selected:", best["neighbor"])

    if return_neighbor:
        return best["tangent"], best["neighbor"]

    return best["tangent"]

def score_reconnection_candidate(G, source, target, forbidden_nodes=None):
    # Work to be done here. The tangent is not very stable sometimes.
    forbidden_nodes = set() if forbidden_nodes is None else set(forbidden_nodes)
    if target == source or target in forbidden_nodes:
        return None

    p0 = np.asarray(G.nodes[source]["xyz"], dtype=np.float64)
    p1 = np.asarray(G.nodes[target]["xyz"], dtype=np.float64)
    reconnect_vec = p1 - p0
    dist = float(np.linalg.norm(reconnect_vec))
    dxy = float(np.linalg.norm(reconnect_vec[:2]))
    dz = float(abs(reconnect_vec[2]))

    if dist < 1e-8:
        return None

    raw_t0 = -estimate_tangent_from_largest_radius_branch(G, source)
    raw_t1 = estimate_tangent_from_largest_radius_branch(G, target)

    if raw_t0 is None or raw_t1 is None:
        return None

    control_pts, simvc_info = optimize_bezier_simvc_nd(
        p0,
        p1,
        raw_t0,
        -raw_t1,
        n_samples=120,
    )

    if control_pts is None:
        return None

    P0, P1, P2, P3 = control_pts

    t = np.linspace(0.0, 1.0, 32)
    curve_pts = bezier_cubic(P0, P1, P2, P3, t)

    # Actual bridge endpoint tangents.
    bridge_start_tangent = curve_pts[1] - curve_pts[0]
    bridge_end_tangent = curve_pts[-1] - curve_pts[-2]

    a1 = angle_deg(raw_t0, bridge_start_tangent)
    a2 = angle_deg(raw_t1, bridge_end_tangent)

    angle_penalty = (a1 + a2) / 180.0

    curvature_penalty = float(simvc_info["fun"])

    if curvature_penalty is None or np.isnan(curvature_penalty):
        curvature_penalty = second_derivative_smoothness(curve_pts)

    return {
        # "node": target,
        "source": source,
        "target": target,

        "distance_xy": dxy,
        "distance": dist,
        "distance_z": dz,

        "angle_penalty": float(angle_penalty),
        "curvature_penalty": float(curvature_penalty),
        "angle0": a1,
        "angle1": a2,

        "bezier_c1": float(simvc_info.get("c1", np.nan)),
        "bezier_c2": float(simvc_info.get("c2", np.nan)),
        "bezier_success": bool(simvc_info.get("success", False)),

        # Store curve/control points so bridge creation can reuse them.
        "bridge_points_xyz": np.asarray(curve_pts, dtype=np.float32),
        "bezier_control_points": tuple(
            np.asarray(p, dtype=np.float32) for p in control_pts
        )
    }

def normalize_and_score_candidates(
    candidates,
    xy_distance_weight=1.0,
    z_distance_weight=1.0,
    angle_weight=1.0,
    curvature_weight=1.0,
    eps=1e-8,
):
    """
    Normalize candidate terms by the average value of each term
    across the current candidate set.

    After this:
        1.0 means average
        <1.0 means better than average
        >1.0 means worse than average
    """
    candidates = [c for c in candidates if c is not None]

    if len(candidates) == 0:
        return []

    avg_xy = float(np.mean([c["distance_xy"] for c in candidates]))
    avg_z = float(np.mean([c["distance_z"] for c in candidates]))
    avg_angle = float(np.mean([c["angle_penalty"] for c in candidates]))
    avg_curvature = float(np.mean([c["curvature_penalty"] for c in candidates]))

    avg_xy = max(avg_xy, eps)
    avg_z = max(avg_z, eps)
    avg_angle = max(avg_angle, eps)
    avg_curvature = max(avg_curvature, eps)

    for c in candidates:
        c["distance_xy_norm"] = float(c["distance_xy"] / avg_xy)
        c["distance_z_norm"] = float(c["distance_z"] / avg_z)
        c["angle_norm"] = float(c["angle_penalty"] / avg_angle)
        c["curvature_norm"] = float(c["curvature_penalty"] / avg_curvature)

        c["score"] = float(
            xy_distance_weight * c["distance_xy_norm"]
            + z_distance_weight * c["distance_z_norm"]
            + angle_weight * c["angle_norm"]
            + curvature_weight * c["curvature_norm"]
        )

    candidates.sort(key=lambda c: c["score"])

    return candidates


def add_curved_repair_edge(G, source, target, source_exclude=None, target_exclude=None, best=None, n_point=320, forbidden_edges=None):
    p0 = np.asarray(G.nodes[source]["xyz"], dtype=np.float32)
    p1 = np.asarray(G.nodes[target]["xyz"], dtype=np.float32)

    repair_curve = None

    if best is not None:
        if "bridge_points_xyz" in best and best["bridge_points_xyz"] is not None:
            repair_curve = np.asarray(best["bridge_points_xyz"], dtype=np.float32)

    if repair_curve is None:
            raw_t0 = -estimate_tangent_from_largest_radius_branch(G, source)
            raw_t1 = estimate_tangent_from_largest_radius_branch(G, target)

            control_pts, simvc_info = optimize_bezier_simvc_nd(
                p0,
                p1,
                raw_t0,
                -raw_t1,
                n_samples=120,
                x0_fraction=0.35,
                min_handle_fraction=0.03,
                max_handle_fraction=1.50,
            )
            if control_pts is None:
                print(
                    f"SIMVC Bézier failed for repair edge {source} -> {target}. "
                    f"Falling back to straight segment."
                )
                repair_curve = np.vstack([p0, p1]).astype(np.float32)
                simvc_info = {
                    "success": False,
                    "fun": np.nan,
                    "c1": np.nan,
                    "c2": np.nan,
                }
            else:
                repair_curve = sample_bezier_bridge(
                    control_pts,
                    n_points=n_point,
                ).astype(np.float32)

    if len(repair_curve) < 2:
        repair_curve = np.vstack([p0, p1]).astype(np.float32)

    repair_curve = np.asarray(repair_curve, dtype=np.float32).copy()
    repair_curve[0] = p0
    repair_curve[-1] = p1

    # smoothness = second_derivative_smoothness(repair_curve)

    r0 = float(G.nodes[source].get("radius_mm", 1.5))
    r1 = float(G.nodes[target].get("radius_mm", 1.5))
    repair_radii = np.linspace(r0, r1, len(repair_curve)).astype(np.float32)

    G.add_edge(
        source,
        target,
        points_xyz=repair_curve.astype(np.float32),
        radius_mm=repair_radii,
        active=True,
        repaired=True,
        permanent=True,
        curved_repair=True,
        score=best["score"],
        component_id=REPAIR_EDGES,
        # forbidden_edges=forbidden_edges,
    )

def get_forbidden_repair_targets(G, source):
    """
    Persistent list of target nodes that this source node should not
    reconnect to again.
    """
    if source not in G:
        return set()

    return set(G.nodes[source].get("forbidden_repair_targets", []))

def add_forbidden_repair_target(G, source, target):
    """
    After repairing source -> target, store target in source's forbidden list.
    This is one-directional by design.
    """
    if source not in G:
        return

    forbidden = get_forbidden_repair_targets(G, source)
    forbidden.add(int(target))

    G.nodes[source]["forbidden_repair_targets"] = sorted(forbidden)


def is_forbidden_repair_target(G, source, target):
    """
    Return True if source has already repaired to target before.
    """
    forbidden = get_forbidden_repair_targets(G, source)
    return int(target) in forbidden

def reconnect_best_candidate_between_components(
    G,
    source_comp,
    largest_comp,
    source_exclude=None,
    target_exclude=None,
    max_candidates=200,
    max_distance_mm=80.0,
    max_angle_deg=100.0,
):

    source_comp = set(source_comp)
    largest_comp = set(largest_comp)
    if (len(source_comp) == 1):
        source_endpoints = list(source_comp)
    else:
        source_endpoints = largest_radius_endpoints_in_component(G, source_comp, top_k=1)
    target_endpoints = [n for n in largest_comp if (G.degree[n] == 1 or G.degree[n] >= 3)]

    if len(source_endpoints) == 0:
        print("Skipping component: no degree-1 endpoints.")
        return None

    if len(target_endpoints) == 0:
        print("Largest component has no degree-1 endpoints.")
        return None

    scores = []

    for source in source_endpoints:
        ps = np.asarray(G.nodes[source]["xyz"], dtype=np.float64)

        nearest_targets = sorted(
            target_endpoints,
            key=lambda target: np.linalg.norm(
                np.asarray(G.nodes[target]["xyz"], dtype=np.float64) - ps
            ),
        )[:max_candidates]

        for target in nearest_targets:
            if is_forbidden_repair_target(G, source, target):
                print(f"Skipping forbidden repair target {source} -> {target}")
                continue
            if nx.has_path(G, source, target):
                continue

            s = score_reconnection_candidate(
                G,
                source=source,
                target=target
            )

            if s is None:
                continue

            if s["distance"] > max_distance_mm:
                continue

            # if s.get("angle0", 0.0) > max_angle_deg:
            #     continue

            # if s.get("angle1", 0.0) > max_angle_deg:
            #     continue

            scores.append(s)
    if len(scores) == 0:
        print("No valid bridge found for this component.")
        return None
    scored_candidates = normalize_and_score_candidates(
        scores,
        xy_distance_weight=XY_DISTANCE_WEIGHT,
        z_distance_weight=Z_DISTANCE_WEIGHT,
        angle_weight=ANGLE_WEIGHT,
        curvature_weight=CURVATURE_WEIGHT
    )
    # for c in scored_candidates[:3]:
    #     print(
    #         f"candidate {c['source']}->{c['target']} | "
    #         f"score={c['score']:.3f}, "
    #         f"dist={c['distance']:.3f}, distance_xy={c['distance_xy']:.3f}, distance_z={c['distance_z']:.3f}, "
    #         f"angle={c['angle_penalty']:.3f}, angle_norm={c['angle_norm']:.3f}, "
    #         f"curv={c['curvature_penalty']:.6f}, curv_norm={c['curvature_norm']:.3f}, "
    #         f"a1={c['angle0']:.1f}, a2={c['angle1']:.1f}"
    #     )

    best = scored_candidates[0] if len(scored_candidates) > 0 else None

    add_curved_repair_edge(
        G,
        source=best["source"],
        target=best["target"],
        source_exclude=source_exclude,
        target_exclude=target_exclude,    
        best=best)
    add_forbidden_repair_target(G, best["source"], best["target"])

    G.edges[best["source"], best["target"]]["component_id"] = REPAIR_EDGES

    return best

def preprocess_reconnect_components_to_largest(
    G,
    max_candidates=200,
    max_distance_mm=25.0,
    max_angle_deg=80.0,
    max_curvature=None,
    min_component_nodes=2,
):
    """
    Preprocess step:
    - assign original component ids
    - reconnect each smaller component to the largest component
    - preserve original component colors via node/edge component_id
    """
    original_components = assign_original_component_ids(G)

    if len(original_components) <= 1:
        print("Graph already has one component. No preprocessing reconnection needed.")
        return []

    largest_comp = set(original_components[0])
    repairs = []

    # print("Preprocessing component reconnection...")

    for comp_id, comp in enumerate(original_components[1:], start=1):
        comp = set(comp)

        if len(comp) < min_component_nodes:
            print(f"Skipping component {comp_id}: too small, nodes={len(comp)}")
            continue

        # print(
        #     f"Trying to reconnect component {comp_id} "
        #     f"nodes={len(comp)} to largest component..."
        # )

        best = reconnect_best_candidate_between_components(
            G,
            source_comp=comp,
            largest_comp=largest_comp
        )

        if best is not None:
            repairs.append(best)

            # After adding the bridge, this component is now connected to largest.
            # Expand largest_comp so later components can also connect to the
            # already-repaired main network.
            largest_comp = set(nx.node_connected_component(G, next(iter(largest_comp))))


    for u, v, data in G.edges(data=True):
        if data.get("active", True):
            add_forbidden_repair_target(G, u, v)
            add_forbidden_repair_target(G, v, u)

    # print("Preprocessing reconnection complete.")
    # print("Repairs added:", len(repairs))
    # print("Current connected components:", nx.number_connected_components(G))

    return repairs

def reconnect_best_candidate_within_component(
    G,
    root_node,
    u,
    v,
    max_candidates=200,
):

    if not G.has_edge(u, v):
        raise RuntimeError(f"Clicked edge ({u}, {v}) no longer exists.")

    old_edge_data = copy.deepcopy(G.edges[u, v])

    print(f"Deleting clicked edge ({u}, {v})")

    G.remove_edge(u, v)

    # If u and v are still connected, deleting the edge did not create
    # a clean between-component problem.
    if nx.has_path(G, u, v):
        G.add_edge(u, v, **old_edge_data)
        raise RuntimeError(
            f"Deleting edge ({u}, {v}) did not split the graph. "
            "This edge is probably part of a loop."
        )
    comp_u = set(nx.node_connected_component(G, u))
    comp_v = set(nx.node_connected_component(G, v))

    if len(comp_u) >= len(comp_v):
        main_comp = comp_u
        source_comp = {v}

        # source_node = v
    else:
        # if (len(comp_u) == 0):
        #     return
        main_comp = comp_v
        source_comp = {u}

        # source_node = u
    try:
        best = reconnect_best_candidate_between_components(
            G,
            source_comp=source_comp,
            largest_comp=main_comp,
            max_candidates=max_candidates,
            # forbidden_edges={(u, v)}
        )

        if best is None:
            raise RuntimeError("No valid reconnection candidate was found.")
    except Exception as e:
        raise RuntimeError(f"Error occurred while finding reconnection candidate: {e}")

    # Post-condition: the graph should still be cycle-free.
    if not nx.is_forest(G):
        G.remove_edge(best['source'], best['target'])
        raise RuntimeError(
            f"Repair {best['source']}->{best['target']} was reverted because it created a loop."
        )

    return best['source'], best['target'], best['score']

# ============================================================
# Visualization
# ============================================================

def graph_edges_to_polydata(
    G,
    repaired_only=False,
    normal_only=False,
    edge_list=None,
):
    points = []
    lines = []
    point_offset = 0

    if edge_list is None:
        edge_iter = G.edges(data=True)
    else:
        edge_iter = [
            (u, v, G.edges[u, v])
            for u, v in edge_list
            if G.has_edge(u, v)
        ]

    for u, v, data in edge_iter:
        if not data.get("active", True):
            continue

        if repaired_only and not data.get("repaired", False):
            continue

        if normal_only and data.get("repaired", False):
            continue

        pts = data.get("points_xyz", None)

        if pts is None:
            continue

        pts = np.asarray(pts, dtype=np.float32)

        if len(pts) < 2:
            continue

        n = len(pts)
        points.append(pts)

        lines.append(
            np.concatenate([
                np.array([n], dtype=np.int64),
                np.arange(point_offset, point_offset + n, dtype=np.int64),
            ])
        )

        point_offset += n

    poly = pv.PolyData()

    if len(points) == 0:
        return poly

    poly.points = np.vstack(points)
    poly.lines = np.concatenate(lines)

    return poly

def collapse_nodes_to_super_root(G, nodes_to_collapse, root_xyz=None):
    nodes_to_collapse = set(nodes_to_collapse)

    if not nodes_to_collapse:
        raise RuntimeError("No nodes selected for collapse.")

    if root_xyz is None:
        root_xyz = np.mean(
            [G.nodes[n]["xyz"] for n in nodes_to_collapse],
            axis=0,
        ).astype(np.float32)

    new_root = max(G.nodes) + 1

    G.add_node(
        new_root,
        xyz=root_xyz,
        ijk=None,
        radius_mm=np.mean([
            G.nodes[n].get("radius_mm", 1.5)
            for n in nodes_to_collapse
        ]),
        degree=1,
        is_super_root=True,
    )

    external_neighbors = set()

    for n in nodes_to_collapse:
        for nb in list(G.neighbors(n)):
            if nb not in nodes_to_collapse:
                external_neighbors.add(nb)

    for nb in external_neighbors:
        p0 = root_xyz
        p1 = np.asarray(G.nodes[nb]["xyz"], dtype=np.float32)

        G.add_edge(
            new_root,
            nb,
            points_xyz=np.vstack([p0, p1]).astype(np.float32),
            radius_mm=np.array([
                G.nodes[new_root].get("radius_mm", 1.5),
                G.nodes[nb].get("radius_mm", 1.5),
            ], dtype=np.float32),
            active=True,
            collapsed_root_edge=True,
        )

    G.remove_nodes_from(nodes_to_collapse)

    return G, new_root

def collapse_main_trachea_region(G, z_min):
    """
    Collapse all nodes above z_min into one super-root.
    Use z_min below both noisy trachea bulks but above the first bifurcation.
    """
    trachea_nodes = [
        n for n in G.nodes
        if float(G.nodes[n]["xyz"][2]) >= z_min
    ]

    if not trachea_nodes:
        raise RuntimeError("No trachea nodes found. Lower z_min.")

    G, new_root = collapse_nodes_to_super_root(G, trachea_nodes)

    # Root should now be this single collapsed trachea node
    return G, new_root


# ============================================================
# 3D airway mesh rendering helpers from the CT-rendering file
# ============================================================

def mask_to_mesh(mask, spacing):
    """
    Convert the extracted airway binary mask to a smoothed PyVista surface mesh.

    This is the same rendering idea as your 3D CT/airway rendering file:
    mask -> marching cubes -> PolyData -> clean -> smooth.
    """
    mask = mask.astype(np.uint8)

    if mask.sum() == 0 or mask.min() == mask.max():
        raise RuntimeError("Invalid mask for meshing.")

    verts, faces, _, _ = measure.marching_cubes(
        mask,
        level=0.5,
        spacing=spacing
    )

    faces_pv = np.hstack([
        np.full((faces.shape[0], 1), 3),
        faces
    ]).astype(np.int64)

    mesh = pv.PolyData(verts, faces_pv)
    mesh = mesh.clean()
    mesh = mesh.smooth(n_iter=20, relaxation_factor=0.05)

    return mesh


def make_polyline_polydata(points):
    points = np.asarray(points, dtype=np.float32)

    poly = pv.PolyData()
    poly.points = points

    if len(points) >= 2:
        poly.lines = np.concatenate([
            np.array([len(points)], dtype=np.int64),
            np.arange(len(points), dtype=np.int64)
        ])

    return poly


def _orthonormal_frame_from_tangent(tangent):
    tangent = np.asarray(tangent, dtype=np.float64)
    tangent = tangent / max(np.linalg.norm(tangent), 1e-12)

    helper = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(np.dot(tangent, helper)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    n1 = np.cross(tangent, helper)
    n1 = n1 / max(np.linalg.norm(n1), 1e-12)
    n2 = np.cross(tangent, n1)
    n2 = n2 / max(np.linalg.norm(n2), 1e-12)

    return n1, n2


def centerline_radius_to_tube_mesh(points, radii, n_sides=16):
    """
    Build a tapered tube mesh around a repaired centerline.

    Unlike pv.PolyData.tube(radius=...), this explicitly supports a different
    radius at each centerline sample, which is what we need after reconnection.
    """
    points = np.asarray(points, dtype=np.float32)
    radii = np.asarray(radii, dtype=np.float32)

    if len(points) < 2:
        return pv.PolyData()

    if len(radii) != len(points):
        r0 = float(radii[0]) if len(radii) > 0 else 1.5
        radii = np.ones(len(points), dtype=np.float32) * r0

    vertices = []
    faces = []

    angles = np.linspace(0.0, 2.0 * np.pi, n_sides, endpoint=False)

    prev_n1 = None
    prev_n2 = None

    for i, p in enumerate(points):
        if i == 0:
            tangent = points[1] - points[0]
        elif i == len(points) - 1:
            tangent = points[-1] - points[-2]
        else:
            tangent = points[i + 1] - points[i - 1]

        if prev_n1 is None:
            n1, n2 = _orthonormal_frame_from_tangent(tangent)
        else:
            # Keep frames stable enough for visualization.
            n1, n2 = prev_n1, prev_n2

        prev_n1, prev_n2 = n1, n2

        r = max(float(radii[i]), 0.05)

        for a in angles:
            vertices.append(p + r * (np.cos(a) * n1 + np.sin(a) * n2))

    vertices = np.asarray(vertices, dtype=np.float32)

    for i in range(len(points) - 1):
        for j in range(n_sides):
            a = i * n_sides + j
            b = i * n_sides + ((j + 1) % n_sides)
            c = (i + 1) * n_sides + ((j + 1) % n_sides)
            d = (i + 1) * n_sides + j
            faces.extend([4, a, b, c, d])

    return pv.PolyData(vertices, np.asarray(faces, dtype=np.int64)).clean()


def graph_edges_to_tube_mesh(G, repaired_only=False, normal_only=False, n_sides=16):
    """
    Convert graph centerline edges into tube meshes using each edge's radius_mm.

    This is the INVANER-style graph view: the graph itself is rendered as
    radius-aware tubes, independently of the original marching-cubes voxel mesh.
    """
    tubes = []

    for u, v, data in G.edges(data=True):
        if not data.get("active", True):
            continue
        if repaired_only and not data.get("repaired", False):
            continue
        if normal_only and data.get("repaired", False):
            continue

        pts = np.asarray(data.get("points_xyz", []), dtype=np.float32)
        if len(pts) < 2:
            pts = np.vstack([G.nodes[u]["xyz"], G.nodes[v]["xyz"]]).astype(np.float32)

        radii = np.asarray(data.get("radius_mm", []), dtype=np.float32)
        if len(radii) != len(pts):
            r0 = float(G.nodes[u].get("radius_mm", 1.5))
            r1 = float(G.nodes[v].get("radius_mm", 1.5))
            radii = np.linspace(r0, r1, len(pts)).astype(np.float32)

        tube = centerline_radius_to_tube_mesh(pts, radii, n_sides=n_sides)
        if tube.n_points > 0:
            tubes.append(tube)

    if not tubes:
        return pv.PolyData()

    combined = tubes[0]
    for tube in tubes[1:]:
        combined = combined.merge(tube, merge_points=False)

    return combined.clean()

def catmull_rom_spline(points, samples_per_segment=12):
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) < 2:
        return pts

    if len(pts) == 2:
        return resample_centerline_for_tube(pts, samples_per_segment)

    out = []

    for i in range(len(pts) - 1):
        p0 = pts[max(i - 1, 0)]
        p1 = pts[i]
        p2 = pts[i + 1]
        p3 = pts[min(i + 2, len(pts) - 1)]

        for t in np.linspace(0.0, 1.0, samples_per_segment, endpoint=False):
            t2 = t * t
            t3 = t2 * t

            p = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * t
                + (2.0*p0 - 5.0*p1 + 4.0*p2 - p3) * t2
                + (-p0 + 3.0*p1 - 3.0*p2 + p3) * t3
            )

            out.append(p)

    out.append(pts[-1])
    return np.asarray(out, dtype=np.float32)

def resample_centerline_for_tube(points, samples_per_segment=6):
    pts = np.asarray(points, dtype=np.float32)

    if len(pts) < 2:
        return pts

    out = []

    for a, b in zip(pts[:-1], pts[1:]):
        for t in np.linspace(
            0.0,
            1.0,
            samples_per_segment,
            endpoint=False,
        ):
            out.append((1.0 - t) * a + t * b)

    out.append(pts[-1])

    return np.asarray(out, dtype=np.float32)
# ============================================================
# Dual interface: skeleton editor + 3D CT/airway rendering
# ============================================================

def visualize_graph_and_airway_linked(G, root_node,
    source_node,
    candidate_nodes,
    airway_mesh,
    forbidden_nodes=None,
    xy_distance_weight=3.0,
    z_distance_weight=0.5,
    angle_weight=25.0,
    curvature_weight=35.0,
    left_clicking = False):
    """
    Three linked views:
        1. Skeleton editor view: graph centerlines/nodes for picking and repair.
        2. Voxel/CT view: original marching-cubes airway surface from the binary mask.
        3. Graph tube view: INVANER-style tube rendering reconstructed from G.

    Repairs are committed to G, then the skeleton and graph tube views are rebuilt
    from the current graph. The voxel view stays as the original extracted airway surface.
    """
    import numpy as np
    import pyvista as pv
    import networkx as nx

    plotter = pv.Plotter(shape=(1, 2), window_size=(1800, 850))
    plotter.set_background("white")

    # Skeleton panel actor
    edge_actor = None
    repaired_actor = None
    node_actor = None
    text_actor = None
    component_actor = []
    component_node_actor = []
    graph_base_actor = None
    graph_repair_actor = None
    graph_node_actor = None
    graph_surface_actor = None
    selected_actor = None

    # Graph tube panel actor
    graph_base_tube_actor = None
    graph_repair_tube_actor = None
    graph_repair_centerline_actor = None
    graph_text_actor = None

    state = {
        "last_cut": None,
        "last_reconnect": None,
        "last_score": None
    }

    def edge_points_to_polydata(points_xyz):
        """
        Convert one edge polyline into PyVista PolyData.
        """
        pts = np.asarray(points_xyz, dtype=np.float32)

        if pts.ndim != 2 or len(pts) < 2:
            return pv.PolyData()

        poly = pv.PolyData()
        poly.points = pts
        poly.lines = np.concatenate([
            np.array([len(pts)], dtype=np.int64),
            np.arange(len(pts), dtype=np.int64),
        ])

        return poly


    def copy_edge_points_xyz(G, u, v):
        """
        Copy the selected edge geometry before the graph is modified.
        """
        data = G.edges[u, v]
        pts = data.get("points_xyz", None)

        if pts is None or len(pts) < 2:
            pts = np.vstack([
                np.asarray(G.nodes[u]["xyz"], dtype=np.float32),
                np.asarray(G.nodes[v]["xyz"], dtype=np.float32),
            ])

        return np.asarray(pts, dtype=np.float32).copy()


    def highlight_edge_in_original_graph(edge_points_xyz):
        """
        Draw the picked skeleton edge as an orange overlay in the original graph panel.
        """
        nonlocal selected_actor

        plotter.subplot(0,1)

        if selected_actor is not None:
            try:
                plotter.remove_actor(selected_actor)
            except Exception:
                pass
            selected_actor = None

        edge_poly = edge_points_to_polydata(edge_points_xyz)

        if edge_poly.n_points > 0:
            selected_actor = plotter.add_mesh(
                edge_poly,
                color="orange",
                line_width=10,
                render_lines_as_tubes=True,
                pickable=False,
            )

        plotter.render()

    # ----------------------------
    # Middle panel: original voxel/CT-derived airway surface
    # ----------------------------
    plotter.subplot(0, 1)
    plotter.add_mesh(
        airway_mesh,
        color="lightyellow",
        smooth_shading=True,
        opacity=0.45,
    )
    plotter.add_axes()

    def graph_edges_to_tube_mesh(
        G,
        repaired_only=False,
        normal_only=False,
        n_sides=18,
    ):
        meshes = []

        for u, v, data in G.edges(data=True):

            if not data.get("active", True):
                continue

            if repaired_only and not data.get("repaired", False):
                continue

            if normal_only and data.get("repaired", False):
                continue

            pts = np.asarray(
                data.get("points_xyz", []),
                dtype=np.float32,
            )

            if len(pts) < 2:
                # pts = np.vstack([
                #     G.nodes[u]["xyz"],
                #     G.nodes[v]["xyz"],
                # ]).astype(np.float32)
                print(f"Skipping edge ({u}, {v}) because it has no valid points_xyz")
                continue

            radii = np.asarray(
                data.get("radius_mm", []),
                dtype=np.float32,
            )

            if len(radii) != len(pts):
                r0 = float(G.nodes[u].get("radius_mm", 1.5))
                r1 = float(G.nodes[v].get("radius_mm", 1.5))

                radii = np.linspace(
                    r0,
                    r1,
                    len(pts),
                ).astype(np.float32)

            # pts_dense = resample_centerline_for_tube(
            #     pts,
            #     samples_per_segment=6,
            # )
            pts_dense = catmull_rom_spline(
                pts,
                samples_per_segment=12,
            )

            radii_dense = np.interp(
                np.linspace(0.0, 1.0, len(pts_dense)),
                np.linspace(0.0, 1.0, len(radii)),
                radii,
            ).astype(np.float32)

            tube = centerline_radius_to_tube_mesh(
                pts_dense,
                radii_dense,
                n_sides=n_sides,
            )

            if tube.n_points > 0:
                meshes.append(tube)

        if not meshes:
            return pv.PolyData()

        merged = meshes[0]

        for m in meshes[1:]:
            merged = merged.merge(m)

        return merged.clean()


    def refresh_skeleton_panel():
        nonlocal edge_actor, repaired_actor, node_actor, text_actor
        nonlocal component_actor, component_node_actor

        plotter.subplot(0, 0)

        # Remove old single-color actor
        for actor in (edge_actor, repaired_actor, node_actor, text_actor):
            if actor is not None:
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass

        # Remove old component-colored actor
        for actor in component_actor:
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass

        for actor in component_node_actor:
            try:
                plotter.remove_actor(actor)
            except Exception:
                pass

        component_actor = []
        component_node_actor = []

        # ---------------------------------------------------
        # Draw each connected component in a different color
        # ---------------------------------------------------
        components = list(nx.connected_components(G))
        components = sorted(components, key=len, reverse=True)

        component_colors = {
            0: "olive",          # original main component
            1: "limegreen",    # repair edges
            2: "dodgerblue",   # separate components
            3: "red"
        }

        component_ids = sorted({
            data.get("component_id")
            for _, _, data in G.edges(data=True)
            if isinstance(data.get("component_id"), int)
        })

        for comp_id in component_ids:
            comp_edges = [
                (u, v)
                for u, v, data in G.edges(data=True)
                if data.get("active", True)
                and data.get("component_id") == comp_id
            ]

            edge_poly = graph_edges_to_polydata(
                G,
                edge_list=comp_edges,
                normal_only=False,
            )

            if edge_poly.n_points > 0:
                actor = plotter.add_mesh(
                    edge_poly,
                    color=component_colors.get(comp_id, "gray"),
                    line_width=5 if comp_id == 1 else 4,
                )
                component_actor.append(actor)


        # ---------------------------------------------------
        # Draw repaired edges on top
        # ---------------------------------------------------
        bridge_edges = [
                (u, v)
                for u, v, data in G.edges(data=True)
                if data.get("component_id") == REPAIR_EDGES
                and data.get("active", True)
            ]

        bridge_poly = graph_edges_to_polydata(
            G,
            edge_list=bridge_edges,
        )

        if bridge_poly.n_points > 0:
            repaired_actor = plotter.add_mesh(
                bridge_poly,
                color="green",
                line_width=7,
            )
        else:
            repaired_actor = None
        for comp_id in sorted({
            data.get("component_id")
            for _, data in G.nodes(data=True)
            if data.get("component_id") is not None
        }):
            node_pts = np.array(
                [
                    data["xyz"]
                    for _, data in G.nodes(data=True)
                    if data.get("component_id") == comp_id
                ],
                dtype=np.float32,
            )

            if len(node_pts) == 0:
                continue

            color = component_colors.get(comp_id, "gray")

            actor = plotter.add_points(
                node_pts,
                color=color,
                point_size=7 if comp_id == 0 else 5,
                render_points_as_spheres=True,
            )
            component_node_actor.append(actor)

        plotter.render()

    def refresh_graph_tube_panel():
        nonlocal graph_base_actor
        nonlocal graph_repair_actor
        nonlocal graph_node_actor
        nonlocal graph_surface_actor

        plotter.subplot(0, 2)

        for actor in (
            graph_base_actor,
            graph_repair_actor,
            graph_node_actor,
            graph_surface_actor,
        ):
            if actor is not None:
                try:
                    plotter.remove_actor(actor)
                except Exception:
                    pass

        # ---------------------------------------------------
        # Transparent marching-cubes airway surface
        # (INVANER-like voxel/surface overlay)
        # ---------------------------------------------------
        graph_surface_actor = plotter.add_mesh(
            airway_mesh,
            color="red",
            opacity=0.18,
            smooth_shading=True,
        )

        # ---------------------------------------------------
        # Normal graph tubes
        # ---------------------------------------------------
        base_tubes = graph_edges_to_tube_mesh(
            G,
            normal_only=True,
            n_sides=18,
        )

        if base_tubes.n_points > 0:
            graph_base_actor = plotter.add_mesh(
                base_tubes,
                color="dodgerblue",
                smooth_shading=True,
                opacity=1.0,
            )

        # ---------------------------------------------------
        # Repaired graph tubes
        # ---------------------------------------------------
        repair_tubes = graph_edges_to_tube_mesh(
            G,
            repaired_only=True,
            n_sides=18,
        )

        if repair_tubes.n_points > 0:
            graph_repair_actor = plotter.add_mesh(
                repair_tubes,
                color="lime",
                smooth_shading=True,
                opacity=1.0,
            )

        # ---------------------------------------------------
        # Graph vertices as dark spheres
        # ---------------------------------------------------
        node_centers = []
        node_radii = []

        for n, data in G.nodes(data=True):
            xyz = np.asarray(data["xyz"], dtype=np.float32)

            r = float(data.get("radius_mm", 1.5))
            r = max(r * 1.15, 1.0)

            node_centers.append(xyz)
            node_radii.append(r)

        node_meshes = []

        for c, r in zip(node_centers, node_radii):
            sph = pv.Sphere(
                radius=r,
                center=c,
                theta_resolution=16,
                phi_resolution=16,
            )
            node_meshes.append(sph)

        if node_meshes:
            merged_nodes = node_meshes[0]

            for s in node_meshes[1:]:
                merged_nodes = merged_nodes.merge(s)

            graph_node_actor = plotter.add_mesh(
                merged_nodes,
                color="midnightblue",
                smooth_shading=True,
            )

        plotter.render()

    def on_pick(point, *args):
        if point is None:
            print(f"No point selected: {point}")
            return

        picked = np.asarray(point, dtype=np.float64)
        print(f"Picked point: {picked}")

        edge, d = nearest_edge_to_point(G, picked, active_only=True)
        if edge is None:
            print("No edge near click.")
            return

        u, v = edge
        print(f"Picked edge ({u}, {v}) at distance {d:.3f}")
        picked_edge_points_xyz = copy_edge_points_xyz(G, u, v)
        try:
            dangling, target, best = reconnect_best_candidate_within_component(
                G,
                root_node,
                u,
                v,
            )

        except Exception as exc:
            print(f"Repair failed: {exc}")
            highlight_edge_in_original_graph(picked_edge_points_xyz)
            return

        state["last_cut"] = (u, v)
        state["last_reconnect"] = (dangling, target)
        state["last_score"] = best

        # Repairs are part of G now. Rebuild views that are graph-derived.
        refresh_skeleton_panel()
        highlight_edge_in_original_graph(picked_edge_points_xyz)
        # refresh_graph_tube_panel()

    # Initial draw
    refresh_skeleton_panel()
    # refresh_graph_tube_panel()
    plotter.add_axes()

    # Keep normal mouse drag for camera movement.
    # Press P over the skeleton panel to pick/cut.
    plotter.enable_point_picking(
        callback=on_pick,
        # show_message="Press P over a false skeleton branch to cut and reconnect it",
        use_mesh=False,
        show_point=True,
        color="yellow",
        point_size=12,
        left_clicking=left_clicking,
    )
    plotter.show()


def assign_original_component_ids(G):
    """
    Store the original connected-component id on every node and edge.

    This must be called BEFORE automatic component reconnection.
    """
    components = sorted(
        nx.connected_components(G),
        key=len,
        reverse=True,
    )
    main_component = set(components[0])

    print("Original components:", len(components))

    for comp_id, comp in enumerate(components):
        sub = G.subgraph(comp)
        for u, v, data in sub.edges(data=True):
            if data.get("component_id") is not None:
                continue
            if u in main_component and v in main_component:
                G.edges[u, v]["component_id"] = ORIGINAL_EDGES
            else:
                G.edges[u, v]["component_id"] = SEPARATE_COMPONENTS

        print(
            f"  component {comp_id}: "
            f"nodes={len(comp)}, "
            f"edges={sub.number_of_edges()}"
        )

    return components

# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("input", help="Input CT .nii or .nii.gz")
    parser.add_argument("--input-type", choices = ["ct", "seg"], default="seg")
    parser.add_argument("--seed", type=int, nargs=3, default=None)

    # Keep the skeleton extraction defaults from the interaction file.
    parser.add_argument("--air-low", type=float, default=-1024)
    parser.add_argument("--air-high", type=float, default=-900)

    parser.add_argument(
        "--max-radius-mm",
        type=float,
        default=12.0,
        help="Lower removes lung leakage more aggressively."
    )

    parser.add_argument(
        "--min-voxels",
        type=int,
        default=800,
        help="Remove tiny disconnected objects."
    )

    parser.add_argument(
        "--collapse-trachea-z-min",
        type=float,
        default=280.0,
        help="Collapse main trachea graph nodes above this z into one super-root. "
             "Set to a negative value to disable."
    )


    args = parser.parse_args()

    ct, affine, spacing = load_nii(args.input)
    if (args.input_type == "ct"):
        if args.seed is None:
            print("Finding trachea seed...")
            seed = find_trachea_seed(ct)
        else:
            seed = tuple(args.seed)
            print("Extracting bronchi / airway mask...")
        airway = extract_airway_only(
            ct=ct,
            seed=seed,
            spacing=spacing,
            air_low=args.air_low,
            air_high=args.air_high,
            max_radius_mm=args.max_radius_mm,
            min_voxels=args.min_voxels,
        )
    else:
        airway = extract_airway_from_segmentation(ct, min_voxels=args.min_voxels)

    airway_mesh = mask_to_mesh(airway, spacing)
    radius_map_mm = ndi.distance_transform_edt(
        airway,
        sampling=spacing
    )

    skeleton = skeletonize_mask(airway)

    G, root_node = skeleton_to_graph(
        skeleton=skeleton,
        spacing=spacing,
        radius_map_mm=radius_map_mm,
    )
    G, voronoi_stats = merge_ballmerge_voronoi_node_clusters(
            G,
            root_node=root_node,
            intersection_threshold=0.3,
            candidate_mode="all",
            max_graph_hops=2,
            require_graph_neighborhood=True,
            max_cluster_diameter_mm=40.0,
            max_cluster_diameter_radius_factor=4.0,
            node_radius_mode="mean_incident_edge_median",
            suppress_degree2_after=True,
            verbose=False,
        )
    
    
    G = smooth_graph_edge_polylines(G, iterations=5, alpha=0.35)
    
    if (args.input_type == "ct"):
        if args.collapse_trachea_z_min >= 0:
            G, root_node = collapse_main_trachea_region(
                G,
                z_min=args.collapse_trachea_z_min,
            )

    G, root_node = remove_loops_keep_tree(G, root_node)
    
    if (args.input_type == "seg"):
        print("Before preprocessing:")
        print("Connected components:", nx.number_connected_components(G))
        repairs = preprocess_reconnect_components_to_largest(G)
        print("After preprocessing:")
        print("Connected components:", nx.number_connected_components(G))

    repair_edges = [
        (u, v, data)
        for u, v, data in G.edges(data=True)
        if data.get("component_id") == REPAIR_EDGES
    ]


    # for u, v, data in repair_edges:
    #     pts = data.get("points_xyz", None)
    #     print(
    #         f"repair edge {u}->{v}: "
    #         f"active={data.get('active')}, "
    #         f"component_id={data.get('component_id')}, "
    #         f"n_points={0 if pts is None else len(pts)}, "
    #         f"distance={data.get('distance')}"
    #     )
    visualize_graph_and_airway_linked(G, root_node, airway_mesh=airway_mesh,
        source_node=root_node, left_clicking=False,
        candidate_nodes=list(G.nodes()),
        forbidden_nodes=None,
        xy_distance_weight=0.8,
        z_distance_weight=0.2,
        angle_weight=25.0,
        curvature_weight=35.0,
    )

    print("Done.")

    


if __name__ == "__main__":
    main()
