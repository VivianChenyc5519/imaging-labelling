import copy
import numpy as np
import networkx as nx

ORIGINAL_EDGES = 0
REPAIR_EDGES = 1
SEPARATE_COMPONENTS = 2
PROBLEMATIC_EDGES = 3


# ============================================================
# BallMerge-style Voronoi-ball node cluster merging
# ============================================================

def _polyline_length(points):
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def _edge_points_oriented(G, u, v):
    """
    Return edge polyline oriented from u to v.
    """
    data = G.edges[u, v]
    pts = np.asarray(data.get("points_xyz", []), dtype=np.float64)

    if len(pts) < 2:
        pts = np.vstack([
            np.asarray(G.nodes[u]["xyz"], dtype=np.float64),
            np.asarray(G.nodes[v]["xyz"], dtype=np.float64),
        ])

    pu = np.asarray(G.nodes[u]["xyz"], dtype=np.float64)

    if np.linalg.norm(pts[0] - pu) > np.linalg.norm(pts[-1] - pu):
        pts = pts[::-1]

    return pts


def _edge_radii_oriented(G, u, v, n_points=None):
    """
    Return edge radii aligned with _edge_points_oriented(G, u, v).

    If edge radius samples are missing or length-mismatched, interpolate
    from the node radii.
    """
    pts = _edge_points_oriented(G, u, v)

    if n_points is None:
        n_points = len(pts)

    data = G.edges[u, v]
    radii = np.asarray(data.get("radius_mm", []), dtype=np.float64)

    if len(radii) == n_points:
        raw_pts = np.asarray(data.get("points_xyz", []), dtype=np.float64)

        if len(raw_pts) >= 2:
            pu = np.asarray(G.nodes[u]["xyz"], dtype=np.float64)
            reversed_needed = (
                np.linalg.norm(raw_pts[0] - pu)
                > np.linalg.norm(raw_pts[-1] - pu)
            )
            if reversed_needed:
                radii = radii[::-1]

        return radii

    ru = float(G.nodes[u].get("radius_mm", 1.5))
    rv = float(G.nodes[v].get("radius_mm", 1.5))

    return np.linspace(ru, rv, n_points)


def _edge_median_radius(G, u, v):
    pts = _edge_points_oriented(G, u, v)
    radii = _edge_radii_oriented(G, u, v, n_points=len(pts))

    if len(radii) == 0:
        return 1.5

    return float(np.median(radii))


def _edge_mean_radius(G, u, v):
    pts = _edge_points_oriented(G, u, v)
    radii = _edge_radii_oriented(G, u, v, n_points=len(pts))

    if len(radii) == 0:
        return 1.5

    return float(np.mean(radii))


def _edge_length(G, u, v):
    return _polyline_length(_edge_points_oriented(G, u, v))


def estimate_node_voronoi_ball_radius(
    G,
    n,
    mode="mean_incident_edge_median",
    fallback_radius_mm=1.5,
    active_only=True,
):
    """
    Estimate a node's approximate Voronoi-ball radius from incident edges.

    Recommended:
        mode="mean_incident_edge_median"

    More aggressive:
        mode="max_incident_edge_median"
    """
    incident = []

    for nb in G.neighbors(n):
        if active_only and not G.edges[n, nb].get("active", True):
            continue

        if mode == "mean_incident_edge_median":
            incident.append(_edge_median_radius(G, n, nb))

        elif mode == "mean_incident_edge_mean":
            incident.append(_edge_mean_radius(G, n, nb))

        elif mode == "max_incident_edge_median":
            incident.append(_edge_median_radius(G, n, nb))

        else:
            raise ValueError(f"Unknown node radius mode: {mode}")

    if len(incident) == 0:
        return float(G.nodes[n].get("radius_mm", fallback_radius_mm))

    if mode == "max_incident_edge_median":
        return float(np.max(incident))

    return float(np.mean(incident))


def ballmerge_intersection_ratio(r0, r1, d, clamp=True, no_overlap_value=0.0):
    """
    BallMerge intersection ratio, computed directly from the paper:

        ir(B0, B1) = max(
            (r0 + r1 - d) / r0,
            (r0 + r1 - d) / r1
        )

    r0, r1:
        Voronoi-ball radii. In your case, each node radius is the
        average radius of the edges connected to that node.

    d:
        Euclidean distance between the two node centers.

    Notes
    -----
    Do not simplify this to division by min(r0, r1) in code.
    The explicit max form is clearer and safer when:
        - r0 != r1
        - the balls do not overlap
        - radii are approximate graph-derived radii, not exact Voronoi radii
    """
    r0 = float(r0)
    r1 = float(r1)
    d = float(d)

    if r0 <= 1e-8 or r1 <= 1e-8:
        return 0.0

    overlap_depth = r0 + r1 - d

    # No geometric overlap.
    if overlap_depth <= 0.0:
        return float(no_overlap_value)

    ir0 = overlap_depth / r0
    ir1 = overlap_depth / r1
    ir = max(ir0, ir1)

    if clamp:
        ir = max(0.0, min(2.0, ir))

    return float(ir)


def _select_candidate_nodes_for_ballmerge(
    G,
    candidate_mode="all",
    protected_nodes=None,
):
    """
    candidate_mode:

    "all":
        Use all nodes except protected nodes.
        Best for twitch/noisy local clusters because degree-2 nodes may
        be part of the artifact.

    "non_degree2":
        Use endpoints and junction-like nodes only.

    "junction_like":
        Use only degree >= 3 nodes.

    "junction_and_leaf":
        Use degree == 1 or degree >= 3.
    """
    if protected_nodes is None:
        protected_nodes = set()
    else:
        protected_nodes = set(protected_nodes)

    nodes = []

    for n in G.nodes:
        if n in protected_nodes:
            continue

        deg = G.degree[n]

        if candidate_mode == "all":
            nodes.append(n)

        elif candidate_mode == "non_degree2":
            if deg != 2:
                nodes.append(n)

        elif candidate_mode == "junction_like":
            if deg >= 3:
                nodes.append(n)

        elif candidate_mode == "junction_and_leaf":
            if deg == 1 or deg >= 3:
                nodes.append(n)

        else:
            raise ValueError(f"Unknown candidate_mode: {candidate_mode}")

    return nodes


def _cluster_diameter_mm(G, cluster):
    pts = np.asarray(
        [G.nodes[n]["xyz"] for n in cluster],
        dtype=np.float64,
    )

    if len(pts) < 2:
        return 0.0

    dmax = 0.0

    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            dmax = max(dmax, float(np.linalg.norm(pts[i] - pts[j])))

    return dmax


def _edge_data_quality_score(data):
    """
    If rewiring creates duplicate edges, keep the stronger one.
    """
    pts = np.asarray(data.get("points_xyz", []), dtype=np.float64)
    length = _polyline_length(pts)

    radii = np.asarray(data.get("radius_mm", []), dtype=np.float64)
    if len(radii) > 0:
        r = float(np.median(radii))
    else:
        r = 1.5

    return 0.05 * length + r


def _add_or_replace_edge(G, u, v, new_data):
    if u == v:
        return

    if not G.has_edge(u, v):
        G.add_edge(u, v, **new_data)
        return

    old_data = G.edges[u, v]

    if _edge_data_quality_score(new_data) > _edge_data_quality_score(old_data):
        G.remove_edge(u, v)
        G.add_edge(u, v, **new_data)


def _choose_largest_radius_representative(
    G,
    cluster,
    node_ball_radius,
    protected_nodes=None,
):
    """
    Keep the node with the largest approximate Voronoi-ball radius.

    If the root/protected node is inside the cluster, keep that instead.
    """
    if protected_nodes is None:
        protected_nodes = set()
    else:
        protected_nodes = set(protected_nodes)

    protected_inside = [n for n in cluster if n in protected_nodes]

    if len(protected_inside) > 0:
        return max(protected_inside, key=lambda n: node_ball_radius.get(n, 0.0))

    return max(cluster, key=lambda n: node_ball_radius.get(n, 0.0))


def _merge_node_cluster_into_rep(
    G,
    cluster,
    rep,
    node_ball_radius,
    cluster_id=None,
    verbose=False,
):
    """
    Merge all nodes in cluster into representative node rep.

    Internal cluster edges are removed.
    External edges are rewired to rep.
    """
    cluster = set(cluster)

    rep_xyz = np.asarray(G.nodes[rep]["xyz"], dtype=np.float64)
    rep_radius = float(node_ball_radius.get(rep, G.nodes[rep].get("radius_mm", 1.5)))

    G.nodes[rep]["voronoi_ball_radius_mm"] = rep_radius
    G.nodes[rep]["ballmerge_cluster_size"] = int(len(cluster))

    if cluster_id is not None:
        G.nodes[rep]["ballmerge_cluster_id"] = int(cluster_id)

    nodes_to_remove = [n for n in cluster if n != rep]

    for n in nodes_to_remove:
        if n not in G:
            continue

        for nb in list(G.neighbors(n)):
            if nb in cluster:
                continue

            old_data = copy.deepcopy(G.edges[n, nb])

            pts = _edge_points_oriented(G, n, nb)
            radii = _edge_radii_oriented(G, n, nb, n_points=len(pts))

            # Re-anchor this external edge to representative node.
            if len(pts) >= 1:
                pts[0] = rep_xyz

            if len(radii) >= 1:
                radii[0] = rep_radius

            old_data["points_xyz"] = pts.astype(np.float32)
            old_data["radius_mm"] = radii.astype(np.float32)
            old_data["active"] = old_data.get("active", True)
            old_data["edge_type"] = old_data.get(
                "edge_type",
                "rewired_after_ballmerge_node_merge",
            )
            old_data["rewired_from_node"] = int(n)
            old_data["rewired_to_rep"] = int(rep)

            _add_or_replace_edge(G, rep, nb, old_data)

        G.remove_node(n)

    if G.has_edge(rep, rep):
        G.remove_edge(rep, rep)

    if verbose:
        print(
            f"[BallMerge node merge] cluster_id={cluster_id}, "
            f"size={len(cluster)}, rep={rep}, "
            f"rep_radius={rep_radius:.2f} mm"
        )


def suppress_degree2_nodes_after_ballmerge(G, protected_nodes=None):
    """
    Optional cleanup.

    A -- J -- B becomes A -------- B.

    This removes pass-through nodes left after cluster merging.
    """
    if protected_nodes is None:
        protected_nodes = set()
    else:
        protected_nodes = set(protected_nodes)

    changed = True

    while changed:
        changed = False

        for n in list(G.nodes):
            if n not in G:
                continue

            if n in protected_nodes:
                continue

            if G.degree[n] != 2:
                continue

            a, b = list(G.neighbors(n))

            if a == b:
                continue

            if G.has_edge(a, b):
                continue

            pts_an = _edge_points_oriented(G, a, n)
            pts_nb = _edge_points_oriented(G, n, b)

            r_an = _edge_radii_oriented(G, a, n, n_points=len(pts_an))
            r_nb = _edge_radii_oriented(G, n, b, n_points=len(pts_nb))

            merged_pts = np.vstack([pts_an, pts_nb[1:]])
            merged_r = np.concatenate([r_an, r_nb[1:]])

            
            G.add_edge(
                a,
                b,
                points_xyz=merged_pts.astype(np.float32),
                radius_mm=merged_r.astype(np.float32),
                active=True,
                component_id=PROBLEMATIC_EDGES,
                edge_type="merged_degree2_after_ballmerge_node_merge",
            )
            # print(f"Add a problematic edge: {G.edges[a,b]["component_id"]}")
            G.remove_node(n)
            changed = True
            break

    return G


def merge_ballmerge_voronoi_node_clusters(
    G,
    root_node=None,
    intersection_threshold=0.75,
    candidate_mode="all",
    max_graph_hops=4,
    require_graph_neighborhood=True,
    max_cluster_diameter_mm=40.0,
    max_cluster_diameter_radius_factor=6.0,
    node_radius_mode="mean_incident_edge_median",
    node_radius_inflation=1.25,
    fallback_radius_mm=1.5,
    suppress_degree2_after=True,
    verbose=True,
    debug_rejected=False,
):
    """
    Merge clustered graph nodes using the BallMerge intersection ratio.

    Each graph node is treated as an approximate Voronoi ball:
        center = node['xyz']
        radius = average/maximum radius of incident edges

    Two nodes are linked into the same merge cluster when:
        ir >= intersection_threshold

    where:
        ir = (r0 + r1 - d) / min(r0, r1)

    This follows the BallMerge idea: deeply overlapping balls are
    likely from the same local region and should be merged.

    Recommended starting values for your airway graph:
        intersection_threshold = 0.75
        candidate_mode = "all"
        max_graph_hops = 4
        node_radius_inflation = 1.25

    Tuning:
        Lower threshold -> more aggressive merging.
        Higher threshold -> more conservative merging.

        0.50 = aggressive
        0.75 = moderate
        1.00 = conservative
        1.25 = very conservative
    """
    protected_nodes = set()
    if root_node is not None:
        protected_nodes.add(root_node)

    stats = {
        "candidate_nodes": 0,
        "ballmerge_pairs": 0,
        "clusters_found": 0,
        "clusters_merged": 0,
        "nodes_removed": 0,
        "clusters_skipped_large": 0,
        "pairs_rejected_low_ir": 0,
        "pairs_rejected_no_overlap": 0,
    }

    # --------------------------------------------------------
    # 1. Estimate approximate Voronoi-ball radius for each node.
    # --------------------------------------------------------
    node_ball_radius = {}

    for n in list(G.nodes):
        r = estimate_node_voronoi_ball_radius(
            G,
            n,
            mode=node_radius_mode,
            fallback_radius_mm=fallback_radius_mm,
            active_only=True,
        )

        r = float(node_radius_inflation * r)

        node_ball_radius[n] = r
        G.nodes[n]["voronoi_ball_radius_mm"] = r

    # --------------------------------------------------------
    # 2. Select candidate nodes.
    # --------------------------------------------------------
    candidates = _select_candidate_nodes_for_ballmerge(
        G,
        candidate_mode=candidate_mode,
        protected_nodes=protected_nodes,
    )

    stats["candidate_nodes"] = len(candidates)

    if len(candidates) < 2:
        return G, stats

    # --------------------------------------------------------
    # 3. Optional graph-hop safety.
    # --------------------------------------------------------
    graph_hop_distance = {}

    if require_graph_neighborhood:
        for n in candidates:
            graph_hop_distance[n] = nx.single_source_shortest_path_length(
                G,
                n,
                cutoff=max_graph_hops,
            )

    # --------------------------------------------------------
    # 4. Build BallMerge overlap graph between graph nodes.
    # --------------------------------------------------------
    H = nx.Graph()
    H.add_nodes_from(candidates)

    for i in range(len(candidates)):
        a = candidates[i]
        ca = np.asarray(G.nodes[a]["xyz"], dtype=np.float64)
        ra = float(node_ball_radius[a])

        if ra <= 0:
            continue

        for j in range(i + 1, len(candidates)):
            b = candidates[j]

            if require_graph_neighborhood:
                if b not in graph_hop_distance.get(a, {}):
                    continue

            cb = np.asarray(G.nodes[b]["xyz"], dtype=np.float64)
            rb = float(node_ball_radius[b])

            if rb <= 0:
                continue

            d = float(np.linalg.norm(ca - cb))

            ir = ballmerge_intersection_ratio(
                ra,
                rb,
                d,
                clamp=True,
            )

            if ir <= 0.0:
                stats["pairs_rejected_no_overlap"] += 1
                continue

            if ir >= intersection_threshold:
                H.add_edge(
                    a,
                    b,
                    intersection_ratio=float(ir),
                    center_distance_mm=float(d),
                    radius_a_mm=float(ra),
                    radius_b_mm=float(rb),
                )
                stats["ballmerge_pairs"] += 1
            else:
                stats["pairs_rejected_low_ir"] += 1

                if debug_rejected and d < 15.0:
                    print(
                        "[BallMerge rejected]",
                        f"a={a}, b={b},",
                        f"d={d:.2f},",
                        f"ra={ra:.2f}, rb={rb:.2f},",
                        f"ir={ir:.3f},",
                        f"threshold={intersection_threshold:.3f}",
                    )

    # --------------------------------------------------------
    # 5. Connected components of H are node clusters.
    # --------------------------------------------------------
    clusters = [
        set(c)
        for c in nx.connected_components(H)
        if len(c) >= 2
    ]

    stats["clusters_found"] = len(clusters)

    if len(clusters) == 0:
        if verbose:
            print("[BallMerge node merge] No node clusters found.")
            print("[BallMerge node merge] Stats:", stats)
        return G, stats

    clusters = sorted(clusters, key=lambda c: len(c), reverse=True)

    # --------------------------------------------------------
    # 6. Merge clusters into largest-radius representative node.
    # --------------------------------------------------------
    cluster_id = 0

    for cluster in clusters:
        cluster = {n for n in cluster if n in G}

        if len(cluster) < 2:
            continue

        diameter = _cluster_diameter_mm(G, cluster)
        max_r = max(float(node_ball_radius.get(n, 0.0)) for n in cluster)

        dynamic_limit = max_cluster_diameter_radius_factor * max_r
        diameter_limit = min(max_cluster_diameter_mm, dynamic_limit)

        if diameter > diameter_limit:
            stats["clusters_skipped_large"] += 1

            if verbose:
                print(
                    "[BallMerge node merge] Skipped large cluster:",
                    f"size={len(cluster)},",
                    f"diameter={diameter:.2f} mm,",
                    f"limit={diameter_limit:.2f} mm",
                )

            continue

        rep = _choose_largest_radius_representative(
            G,
            cluster,
            node_ball_radius,
            protected_nodes=protected_nodes,
        )

        removed_count = len(cluster) - 1

        _merge_node_cluster_into_rep(
            G,
            cluster,
            rep,
            node_ball_radius,
            cluster_id=cluster_id,
            verbose=verbose,
        )

        stats["clusters_merged"] += 1
        stats["nodes_removed"] += removed_count
        cluster_id += 1

    # --------------------------------------------------------
    # 7. Optional degree-2 cleanup.
    # --------------------------------------------------------
    if suppress_degree2_after:
        G = suppress_degree2_nodes_after_ballmerge(
            G,
            protected_nodes=protected_nodes,
        )

    if verbose:
        print("[BallMerge node merge] Stats:", stats)

    return G, stats