# Histogram Confidence Method (HCM)
# Minimal reference implementation

import bisect

class HCM:
    def __init__(self, bins, counts_low, counts_high, edges):
        self.bins = bins                  # number of bins
        self.counts_low = counts_low      # lower confidence bound per bin
        self.counts_high = counts_high    # upper confidence bound per bin
        self.edges = edges                # bin edges

    def update(self, x):
        # find bin in O(1) or O(log n) depending on implementation
        idx = bisect.bisect_right(self.edges, x) - 1
        idx = max(0, min(idx, self.bins - 1))

        # drift detected if outside confidence envelope
        if self.counts_low[idx] <= 0 or self.counts_high[idx] <= 0:
            return False  # malformed baseline

        # True = inside envelope, False = drift
        inside = (self.counts_low[idx] <= 1 <= self.counts_high[idx])
        return inside
