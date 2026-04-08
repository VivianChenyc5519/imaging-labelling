import cv2
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
from scipy.optimize import linear_sum_assignment


def preprocess_mask_opencv(
    vessel_mask,
    min_object_size=5,
    do_open=False,
    do_close=False,
    kernel_size=3,
):
    """
    Clean a binary vessel mask using OpenCV.

    Parameters
    ----------
    vessel_mask : np.ndarray
        2D mask. Nonzero values are treated as vessel pixels.
    min_object_size : int
        Remove connected components smaller than this area.
    do_open : bool
        Apply morphological opening.
    do_close : bool
        Apply morphological closing.
    kernel_size : int
        Kernel size for morphology.

    Returns
    -------
    np.ndarray
        Cleaned binary mask as uint8 with values 0 or 255.
    """
    mask = (vessel_mask > 0).astype(np.uint8) * 255

    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    if do_open:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    if do_close:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

    cleaned = np.zeros_like(mask)
    for label_id in range(1, num_labels):  # skip background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area >= min_object_size:
            cleaned[labels == label_id] = 255

    return cleaned


def extract_dots_from_slice_opencv(
    slice_mask,
    x0,
    x1,
    min_area=2,
    y_mode="mid",
):
    """
    Extract one dot per connected component in a vertical slice.

    Parameters
    ----------
    slice_mask : np.ndarray
        Binary slice mask of shape (H, slice_width), values 0 or 255.
    x0, x1 : int
        Slice start/end columns in the original image, x1 exclusive.
    min_area : int
        Minimum component area to keep.
    y_mode : str
        'mid'      -> y = (y_min + y_max) / 2
        'centroid' -> use connected component centroid

    Returns
    -------
    list[dict]
        Dot information for that slice.
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        slice_mask, connectivity=8
    )

    dots = []
    x_center = 0.5 * (x0 + x1 - 1)

    for label_id in range(1, num_labels):  # skip background
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        left = stats[label_id, cv2.CC_STAT_LEFT]
        top = stats[label_id, cv2.CC_STAT_TOP]
        width = stats[label_id, cv2.CC_STAT_WIDTH]
        height = stats[label_id, cv2.CC_STAT_HEIGHT]

        y_min = top
        y_max = top + height - 1

        if y_mode == "centroid":
            y_dot = float(centroids[label_id][1])  # (cx, cy)
        elif y_mode == "mid":
            y_dot = 0.5 * (y_min + y_max)
        else:
            raise ValueError("y_mode must be 'mid' or 'centroid'")

        dots.append(
            {
                "x": x_center,
                "y": y_dot,
                "area": int(area),
                "height": int(height),
                "bbox": (int(top), int(left), int(top + height), int(left + width)),
                "slice_x0": int(x0),
                "slice_x1": int(x1),
            }
        )

    return dots


def extract_all_dots_opencv(
    vessel_mask,
    slice_width=3,
    step=None,
    min_area=2,
    y_mode="mid",
):
    """
    Extract dots from all vertical slices.

    Parameters
    ----------
    vessel_mask : np.ndarray
        Binary mask, values 0/255 or any nonzero vessel mask.
    slice_width : int
        Width of each vertical slice.
    step : int or None
        Step between slice starts.
        If None, uses step = slice_width.
        If step < slice_width, slices overlap.
    min_area : int
        Minimum connected-component area.
    y_mode : str
        'mid' or 'centroid'.

    Returns
    -------
    observations : list[list[dict]]
    slice_ranges : list[tuple[int, int]]
    all_dots : list[dict]
    """
    if step is None:
        step = slice_width

    mask = (vessel_mask > 0).astype(np.uint8) * 255
    H, W = mask.shape

    observations = []
    slice_ranges = []
    all_dots = []

    for x0 in range(0, W, step):
        x1 = min(x0 + slice_width, W)
        if x0 >= W or x0 == x1:
            continue

        slice_mask = mask[:, x0:x1]

        dots = extract_dots_from_slice_opencv(
            slice_mask=slice_mask,
            x0=x0,
            x1=x1,
            min_area=min_area,
            y_mode=y_mode,
        )

        observations.append(dots)
        slice_ranges.append((x0, x1))
        all_dots.extend(dots)

        if x1 == W:
            break

    return observations, slice_ranges, all_dots


def visualize_mask_and_dots(
    vessel_mask,
    all_dots,
    slice_ranges=None,
    show_slice_boundaries=True,
    boundary_stride=10,
    figsize=(10, 10),
    title="Detected vessel dots over mask",
):
    """
    Show mask with extracted dots overlaid.
    """
    mask = (vessel_mask > 0).astype(np.uint8)

    plt.figure(figsize=figsize)
    plt.imshow(mask, cmap="gray", origin="upper")

    if len(all_dots) > 0:
        xs = [d["x"] for d in all_dots]
        ys = [d["y"] for d in all_dots]
        plt.scatter(xs, ys, s=14, marker="o")

    if show_slice_boundaries and slice_ranges is not None and len(slice_ranges) > 0:
        for i, (x0, x1) in enumerate(slice_ranges):
            if i % boundary_stride == 0:
                plt.axvline(x=x0 - 0.5, linewidth=0.5)
        plt.axvline(x=slice_ranges[-1][1] - 0.5, linewidth=0.5)

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()


def visualize_dots_only(
    image_shape,
    all_dots,
    figsize=(10, 10),
    title="Detected vessel dots only",
):
    """
    Show only extracted dots on a blank canvas.
    """
    H, W = image_shape
    canvas = np.zeros((H, W), dtype=np.uint8)

    plt.figure(figsize=figsize)
    plt.imshow(canvas, cmap="gray", origin="upper")

    if len(all_dots) > 0:
        xs = [d["x"] for d in all_dots]
        ys = [d["y"] for d in all_dots]
        plt.scatter(xs, ys, s=14, marker="o")

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.show()
    
    
# ============================================================
# SECTION 4 — MATCH DOTS BETWEEN ADJACENT SLICES
# ============================================================

def pairwise_cost(dot_a: dict, dot_b: dict,
                  area_weight: float = 0.0,
                  height_weight: float = 0.0) -> float:
    """
    Cost of connecting one dot in slice k to one dot in slice k+1.

    Current cost:
    - vertical difference in y
    - optional area difference
    - optional height difference
    """
    dy = abs(dot_a["y"] - dot_b["y"])
    da = abs(dot_a.get("area", 1) - dot_b.get("area", 1))
    dh = abs(dot_a.get("height", 1) - dot_b.get("height", 1))
    return dy + area_weight * da + height_weight * dh


def match_adjacent_slices(
    dots_a: list,
    dots_b: list,
    max_dy: float = 8.0,
    area_weight: float = 0.0,
    height_weight: float = 0.0,
):
    """
    Connect dots between two adjacent slices using Hungarian assignment.

    Returns
    -------
    matches : list of (index_in_a, index_in_b, cost)
    """
    if len(dots_a) == 0 or len(dots_b) == 0:
        return []

    n = len(dots_a)
    m = len(dots_b)
    BIG = 1e9

    cost_matrix = np.full((n, m), BIG, dtype=np.float64)

    for i, da in enumerate(dots_a):
        for j, db in enumerate(dots_b):
            dy = abs(da["y"] - db["y"])
            if dy <= max_dy:
                cost_matrix[i, j] = pairwise_cost(
                    da, db,
                    area_weight=area_weight,
                    height_weight=height_weight
                )

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    for i, j in zip(row_ind, col_ind):
        cost = cost_matrix[i, j]
        if cost < BIG:
            matches.append((i, j, float(cost)))

    return matches


# ============================================================
# SECTION 5 — UNION-FIND TO GROUP CONNECTED DOTS INTO TRACKS
# ============================================================

class UnionFind:
    """
    Small disjoint-set / union-find structure.
    Used to merge nodes that belong to the same connected track.
    """
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

    def find(self, x):
        px = self.parent[x]
        if px != x:
            self.parent[x] = self.find(px)
        return self.parent[x]

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)

        if ra == rb:
            return

        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


# ============================================================
# SECTION 6 — BUILD THE GRAPH OF CONNECTIVITY ACROSS SLICES
# ============================================================

def build_track_graph(
    observations: list,
    max_dy: float = 8.0,
    area_weight: float = 0.0,
    height_weight: float = 0.0,
):
    """
    Build a graph from the extracted dots.

    Nodes:
        (slice_index, dot_index)

    Edges:
        connections between matched dots in adjacent slices

    Returns
    -------
    nodes : dict
        node_id -> node attributes
    edges : list
        [(node_a, node_b, cost), ...]
    components : list[list[node_id]]
        connected tracks
    node_to_component : dict
        node_id -> connected component index
    """
    nodes = {}
    edges = []
    uf = UnionFind()

    # Create one node for every detected dot
    for s_idx, dots in enumerate(observations):
        for d_idx, dot in enumerate(dots):
            node_id = (s_idx, d_idx)
            nodes[node_id] = {
                "slice": s_idx,
                "dot": d_idx,
                "x": float(dot["x"]),
                "y": float(dot["y"]),
                "area": float(dot.get("area", 1)),
                "height": float(dot.get("height", 1)),
            }
            uf.add(node_id)

    # Connect slice k to slice k+1
    for s_idx in range(len(observations) - 1):
        dots_a = observations[s_idx]
        dots_b = observations[s_idx + 1]

        matches = match_adjacent_slices(
            dots_a,
            dots_b,
            max_dy=max_dy,
            area_weight=area_weight,
            height_weight=height_weight,
        )

        for ia, ib, cost in matches:
            node_a = (s_idx, ia)
            node_b = (s_idx + 1, ib)
            edges.append((node_a, node_b, cost))
            uf.union(node_a, node_b)

    # Extract connected components
    root_to_nodes = {}
    for node_id in nodes:
        root = uf.find(node_id)
        root_to_nodes.setdefault(root, []).append(node_id)

    components = list(root_to_nodes.values())
    node_to_component = {}
    for comp_idx, comp in enumerate(components):
        for node_id in comp:
            node_to_component[node_id] = comp_idx

    return nodes, edges, components, node_to_component


# ============================================================
# SECTION 7 — INTERACTIVE VIEWER
# ============================================================

class InteractiveTrackViewer:
    """
    Interactive matplotlib viewer.

    Click a node:
    - the selected node is marked
    - all nodes in the same track are highlighted
    - all edges in that track are highlighted
    """
    def __init__(
        self,
        vessel_mask: np.ndarray,
        nodes: dict,
        edges: list,
        components: list,
        node_to_component: dict,
        node_size: int = 18,
        click_radius: float = 8.0,
    ):
        self.vessel_mask = vessel_mask
        self.nodes = nodes
        self.edges = edges
        self.components = components
        self.node_to_component = node_to_component
        self.node_size = node_size
        self.click_radius = click_radius

        self.pos = {
            node_id: (attr["x"], attr["y"])
            for node_id, attr in nodes.items()
        }

        self.fig, self.ax = plt.subplots(figsize=(12, 12))
        self.ax.imshow(vessel_mask, cmap="gray", origin="upper")

        # Base edges
        self.base_edge_artists = []
        for node_a, node_b, _ in edges:
            x1, y1 = self.pos[node_a]
            x2, y2 = self.pos[node_b]
            line, = self.ax.plot([x1, x2], [y1, y2], linewidth=0.8, alpha=0.25)
            self.base_edge_artists.append(line)

        # Base nodes
        xs = [self.pos[n][0] for n in nodes]
        ys = [self.pos[n][1] for n in nodes]
        self.base_nodes = self.ax.scatter(xs, ys, s=node_size)

        # Highlight layers
        self.highlight_lines = []
        self.highlight_nodes = self.ax.scatter([], [], s=node_size * 4)
        self.selected_node = self.ax.scatter([], [], s=node_size * 7, marker="x")

        self.ax.set_title("Click a dot to highlight its vessel track")
        self.ax.set_xlabel("x")
        self.ax.set_ylabel("y")

        self.fig.canvas.mpl_connect("button_press_event", self.on_click)

    def nearest_node(self, x, y):
        """
        Find the nearest node to the click.
        """
        if x is None or y is None:
            return None

        best_node = None
        best_d2 = float("inf")

        for node_id, (xn, yn) in self.pos.items():
            d2 = (xn - x) ** 2 + (yn - y) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best_node = node_id

        if np.sqrt(best_d2) <= self.click_radius:
            return best_node
        return None

    def clear_highlight(self):
        for line in self.highlight_lines:
            line.remove()
        self.highlight_lines = []

        self.highlight_nodes.set_offsets(np.empty((0, 2)))
        self.selected_node.set_offsets(np.empty((0, 2)))

    def highlight_track(self, node_id):
        """
        Highlight the connected component containing the clicked node.
        """
        self.clear_highlight()

        comp_idx = self.node_to_component[node_id]
        comp_nodes = self.components[comp_idx]
        comp_set = set(comp_nodes)

        comp_nodes_sorted = sorted(comp_nodes, key=lambda n: self.nodes[n]["slice"])
        comp_xy = np.array([self.pos[n] for n in comp_nodes_sorted], dtype=float)

        if len(comp_xy) > 0:
            self.highlight_nodes.set_offsets(comp_xy)

        for node_a, node_b, _ in self.edges:
            if node_a in comp_set and node_b in comp_set:
                x1, y1 = self.pos[node_a]
                x2, y2 = self.pos[node_b]
                line, = self.ax.plot([x1, x2], [y1, y2], linewidth=2.5)
                self.highlight_lines.append(line)

        sx, sy = self.pos[node_id]
        self.selected_node.set_offsets(np.array([[sx, sy]], dtype=float))

        self.fig.canvas.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax:
            return

        node_id = self.nearest_node(event.xdata, event.ydata)
        if node_id is not None:
            self.highlight_track(node_id)

    def show(self):
        plt.tight_layout()
        plt.show()



# ---------------------------------------------------------
# Example usage
# ---------------------------------------------------------
if __name__ == "__main__":
    # Replace with your vessel mask path
    mask_path = "./1st_manual/21_manual1.gif"

    # Read as grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read image: {mask_path}")

    # Binarize
    vessel_mask = (mask > 0).astype(np.uint8) * 255

    # Optional cleanup
    vessel_mask = preprocess_mask_opencv(
        vessel_mask,
        min_object_size=5,
        do_open=False,
        do_close=False,
        kernel_size=3,
    )

    # Extract dots
    observations, slice_ranges, all_dots = extract_all_dots_opencv(
        vessel_mask,
        slice_width=3,   # thin slice width
        step=1,          # overlapping slices; use step=3 for non-overlapping
        min_area=2,
        y_mode="mid",    # or "centroid"
    )

    print(f"Number of slices: {len(observations)}")
    print(f"Total dots detected: {len(all_dots)}")

    # Visualize mask + dots
    visualize_mask_and_dots(
        vessel_mask,
        all_dots,
        slice_ranges=slice_ranges,
        show_slice_boundaries=True,
        boundary_stride=10,
        figsize=(12, 12),
        title="Detected vessel dots over mask",
    )

    # Visualize dots only
    visualize_dots_only(
        vessel_mask.shape,
        all_dots,
        figsize=(12, 12),
        title="Detected vessel dots only",
    )
    nodes, edges, components, node_to_component = build_track_graph(
        observations,
        max_dy=8.0,
        area_weight=0.0,
        height_weight=0.0,
    )

    print(f"Total nodes: {len(nodes)}")
    print(f"Total edges: {len(edges)}")
    print(f"Total connected tracks: {len(components)}")
    viewer = InteractiveTrackViewer(
    vessel_mask=vessel_mask,
    nodes=nodes,
    edges=edges,
    components=components,
    node_to_component=node_to_component,
    node_size=18,
    click_radius=15.0,
    )

    viewer.show()