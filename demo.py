#!/usr/bin/env python3
# Numpy is for constructing our matrices. See API docs for detan usage.
import numpy as np
from detan.detan import AssignmentAnnealing, assignment_iteration

# This is our pairwise distance matrix. Each entry is the result of some
# pairwise distance metric for each object. Note that the pairwise distance
# metric must be symmetric, so this matrix must be too (hence the triangular
# construction and then transpose).
distances = np.asarray((
    (0.0 , 2.1 , 0.10, 0.85, 0.2 , 0.78),
    (0.0 , 0.0 , 0.92, 0.05, 1.01, 0.01),
    (0.0 , 0.0 , 0.0 , 2.02, 0.15, 0.99),
    (0.0 , 0.0 , 0.0 , 0.0 , 1.30, 0.31),
    (0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 1.05),
    (0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ),
))

distances = distances + distances.T

# It's up to the caller to decide how many groups there are.
groups = 2

# The initial assignment expectations should be random, and must sum to 1 across
# each row. There should be no identical entries in a given row.
initial_assignments = 0.5 + 0.1 * (np.random.random((6,groups)) - 0.5)
row_sum = np.tile(initial_assignments.sum(1), (groups, 1)).T
initial_assignments = initial_assignments/row_sum

# This is the state of our annealling.
annealer = AssignmentAnnealing(assignment_iteration(distances), initial_assignments, 0.73)

# Tolerance for deciding when to lower the temperature. We also need to keep
# track of the old assignments.
tolerance = 1e-6
old_assignments = initial_assignments

# For the sake of simplicity, I've picked an arbitrary number of temperature
# steps.
for _ in range(20):
    # Iterating over the annealer object produces the new assignment
    # expectations.
    for new_assignments in annealer:
        # If none of the assignments change by more than the tolerance, drop the
        # temperature.
        if np.abs(new_assignments - old_assignments).max() < tolerance:
            break
        old_assignments = new_assignments
    # Next temperature.
    annealer.cool()

# The raw assignment expectation values.
print("Raw assignment expectations:")
print(annealer.assignments)

print()

# The "ideal" clustering results.
print("Ideal clustering results:")
print(annealer.assignments > 1e-50)
