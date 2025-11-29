"""
Histogram Confidence Method (HCM)
Minimal reference implementation (SciPy-free).

- Build adaptive baseline histogram from data
- Precompute expected count bounds for a sliding window of size w
- Maintain windowed bin counts at runtime
- Detect drift if any bin count falls outside its confidence envelope
"""

import bisect
import math
from collections import deque
from typing import List, Tuple, Sequence


class HCM:
    def __init__(
        self,
        baseline_data: Sequence[float],
        window_size: int = 500,
        min_per_bin: int = 50,
        max_bins: int = 20,
        confidence: float = 0.99,
    ):
        """
        Initialize HCM from baseline data.

        Args:
            baseline_data: 1D sequence of baseline observations
            window_size: size of monitoring window (w)
            min_per_bin: minimum samples per adaptive bin
            max_bins: maximum number of bins
            confidence: confidence level for bounds (e.g., 0.99)
        """
        data = sorted(baseline_data)
        if len(data) < min_per_bin:
            raise ValueError("Not enough baseline data for requested min_per_bin.")

        self.w = int(window_size)
        self.min_per_bin = int(min_per_bin)
        self.max_bins = int(max_bins)
        self.confidence = float(confidence)

        # Build adaptive histogram from baseline
        self.edges, self.bin_counts = self._build_adaptive_histogram(data)
        self.K = len(self.bin_counts)
        self.N = len(data)

        # Baseline bin probabilities
        self.p = [c / self.N for c in self.bin_counts]

        # Precompute count bounds for each bin in a window of size w
        self.L, self.U = self._compute_count_bounds()

        # Runtime state: sliding window and current bin counts
        self.buffer = deque(maxlen=self.w)  # store last w values
        self.current_counts = [0] * self.K
        self.drift = False

    def _build_adaptive_histogram(
        self, data: List[float]
    ) -> Tuple[List[float], List[int]]:
        """Build adaptive bins with at least min_per_bin samples per bin."""
        n = len(data)
        edges: List[float] = []
        counts: List[int] = []

        i = 0
        while i < n and len(counts) < self.max_bins:
            start = i
            i = min(n, i + self.min_per_bin)

            # If this is the last bin and it's too small, merge with previous
            if i == n and counts and (i - start) < self.min_per_bin * 0.5:
                counts[-1] += (i - start)
                edges[-1] = data[i - 1]
            else:
                counts.append(i - start)
                edges.append(data[i - 1])

        # Leftmost edge just below the minimum value (for bisect)
        left_edge = data[0] - 1e-9
        edges = [left_edge] + edges

        return edges, counts

    def _compute_count_bounds(self) -> Tuple[List[int], List[int]]:
        """
        Compute per-bin lower and upper acceptable counts
        for a window of size w using a normal approximation.

        For bin i with probability p_i:
            E[count_i] = w * p_i
            Var[count_i] = w * p_i * (1 - p_i)
        """
        alpha = 1.0 - self.confidence
        # Two-sided z-score; for 0.99, z ≈ 2.576
        # We'll use an approximation based on the standard normal inverse CDF.
        # To keep this minimal, we hardcode common values.
        if abs(self.confidence - 0.95) < 1e-6:
            z = 1.96
        elif abs(self.confidence - 0.99) < 1e-6:
            z = 2.576
        else:
            # Fallback: simple approximation (fine for most practical uses)
            # z ≈ sqrt(2) * erfc^-1(alpha)
            # We avoid scipy; use a rough approximation via math.erfcinv if present.
            try:
                from math import erfcinv  # type: ignore
                z = math.sqrt(2.0) * erfcinv(alpha)
            except Exception:
                # conservative fallback
                z = 2.576

        L: List[int] = []
        U: List[int] = []

        for p_i in self.p:
            mu = self.w * p_i
            if p_i == 0.0 or mu == 0.0:
                # bin essentially empty under baseline
                L.append(0)
                U.append(0)
                continue

            var = self.w * p_i * (1.0 - p_i)
            sigma = math.sqrt(max(var, 1e-9))
            lower = max(0.0, mu - z * sigma)
            upper = min(self.w, mu + z * sigma)

            L.append(int(math.floor(lower)))
            U.append(int(math.ceil(upper)))

        return L, U

    def _bin_index(self, x: float) -> int:
        """Find histogram bin index for value x."""
        idx = bisect.bisect_right(self.edges, x) - 1
        if idx < 0:
            idx = 0
        elif idx >= self.K:
            idx = self.K - 1
        return idx

    def update(self, x: float) -> bool:
        """
        Process one new observation.

        Returns:
            True if NO drift detected (still in distribution),
            False if drift is detected.
        """
        # Remove oldest from window if full
        if len(self.buffer) == self.w:
            old_x = self.buffer.popleft()
            old_idx = self._bin_index(old_x)
            self.current_counts[old_idx] -= 1

        # Add new value
        self.buffer.append(x)
        idx = self._bin_index(x)
        self.current_counts[idx] += 1

        # Don't evaluate until window is full
        if len(self.buffer) < self.w:
            self.drift = False
            return True

        # Check each bin against its bounds
        drift_now = False
        for i in range(self.K):
            c = self.current_counts[i]
            if c < self.L[i] or c > self.U[i]:
                drift_now = True
                break

        self.drift = drift_now
        return not self.drift

    def is_drift(self) -> bool:
        """Return the latest drift state."""
        return self.drift

    def reset(self) -> None:
        """Reset monitoring state (keep baseline and bounds)."""
        self.buffer.clear()
        self.current_counts = [0] * self.K
        self.drift = False
