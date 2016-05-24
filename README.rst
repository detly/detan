Deterministic Annealing
=======================

``detan`` is a Python 3 library for deterministic annealing, a clustering
algorithm that uses fixed point iteration. It is based on *T. Hofmann and J. M.
Buhmann, Pairwise data clustering by deterministic annealing, IEEE T. Pattern
Anal., 19 (1997), pp. 1–14.*

Installation
------------

You can install directly from this git repository using:

.. code:: text

    pip install git+https://github.com/detly/detan.git

...or you can clone the git repository however you prefer, and do:

.. code:: text

    pip install .

...or:

.. code:: text

    python setup.py install

...from the cloned tree.

Dependencies
~~~~~~~~~~~~

-  numpy

Clustering algorithms
---------------------

Deterministic annealing is a clustering algorithm. So what do I consider a
clustering algorithm in general?

Say you have a collection of ``N`` things. They could be signals, or images, or
something else altogether. Say also that you have some way of measuring the
dissimilarity of each pair of things. This doesn't need to be distance in a
vector space (although it could be) — as long as there's a well defined and
symmetric way to measure the dissimilarity between any two of your things, that
will work.

I will call this dissimilarity the "pairwise distance measure" or just
"distance" from now on, but it's important to remember the note above: it
doesn't have to be a conventional distance (eg. the Euclidean norm). It just has
to be:

-  symmetric: ``distance(A, B) == distance(B, A)``
-  well defined: ``distance(A, B) == distance(C, D)`` whenever ``A == C`` and
   ``B == D``

Finally, say you know that there are ``k`` natural groupings of these objects.
*Clustering* is a way of computing which things go in which group based on their
distances.

All that a clustering algorithm cares about is the dissimilarity measure and the
number of groups. Given an ``N × N`` matrix of your distances (which should be
symmetric), and the number of groups ``k``, clustering will produce an
*assignment matrix* that says which things go into which group.

The assignment matrix is a ``N × k`` matrix. Ideally the assignment matrix will
contain only zeros and ones, where a one in row ``i`` and column ``j`` means
that thing ``i`` goes in group ``j`` (and zero means the opposite). The matrix
should satisfy the following three conditions:

-  binary: each entry is either 0 or 1
-  exclusive, and exhaustive: each row contains exactly one ``1``, because each
   thing can must belong to one and only one group

For example, this matrix:

.. table::

    = = =
    0 1 0
    1 0 0
    1 0 0
    0 0 1
    1 0 0
    0 0 1
    = = =

...means that thing #0 goes in group #1, things #1, #2 and #4 go in group #0,
and things #3 and #5 go in group #2. The order of the groups doesn't matter; any
permutation of the columns would result in an equivalent assignment matrix.

Deterministic annealing is an algorithm that takes such a distance matrix and
*approximates* the assignment matrix.

How deterministic annealing works
---------------------------------

The assignment matrix is the ideal outcome of a clustering algorithm, but it is
not quite what deterministic annealing calculates. Deterministic annealing (or
*DA*) works on the *expectation values* of the assignments. Instead of a matrix
of zeros and ones, DA iterates over a matrix of values *between* zero and one.
DA gradually converges these expectation values towards the optimal zero or one
for that "thing" and "group".

DA works at two levels, which will probably manifest in your code as two nested
loops. The outer loop will gradually lower the temperature parameter, and the
inner loop will update the expectation values.

Updating the expectation values has two parts. First, we calculate "potentials"
from the expectation values and distances (so named because they're an analogue
of potential energy in a physical system). Second, we calculate new expectations
from the potentials and the Lagrangian parameter ``T`` (so named because it's
analogous to temperature in a physical system). This is a form of fixed point
iteration, ie. repeatedly calculating ``x_(n+1) = f(x_n)`` until we decide we've
found the solution.

Limitations and modes of failure
--------------------------------

There is no intrinsically obvious point at which to stop iterating and lower the
temperature, nor is there an obvious point to stop lowering the temperature. It
is entirely dependent on context and the statistics of the distances.

A common criteria for stopping fixed point iteration (the inner loop) is to
calculate the absolute difference between the last value and the current value
and stop when this difference drops below a threshold:

.. code:: python

    for new_assignments in annealer:
        if np.abs(new_assignments - old_assignments).max() < tolerance:
            break
        old_assignments = new_assignments

Deciding when to stop lowering the temperature is more context dependent; the
two ways I've used have been to:

-  have a fixed number of iterations (eg. 20)
-  to look at the maximum distance of the assignment expectations from zero and
   one

This may take some trial and error in your application to determine what works
best.

Another complication arises because of numerical precision. If a thing is
"close" to being in more than one group, the expected assignments could differ
by less than what a computer's numerical precision can express. In this case,
there will be two identical entries in a row of the matrix, and they might both
converge towards ``1``. (Ideally, there would always be a difference, no matter
how slight, and so one entry would end up becoming ``1``).

This can manifest as either a matrix row with two values very close to ``1`` or,
if DA continues to be iterated after this point, ``NaN`` entries in the
assignment matrix. It's up to the caller to detect this kind of failure, and in
my experience, increasing the "cooling" ratio can help. There are functions to
restore previous values when this happens so that you don't lose information.

Finally, it's not part of DA to detect how many groups to use. That decision is
up to the caller.

Usage
-----

The `API documentation <http://detly.github.io/detan/>`__ has details on the
API, but here's a breakdown on how to put calling code together.

First, the imports. We'll use ``numpy`` for putting matrices together. The two
things you'll probably want from ``detan`` are:

.. code:: python

    import numpy as np
    from detan.detan import AssignmentAnnealing, assignment_iteration

Remember how one part of DA is to calculate new expectation values from old
ones? ``detan`` allows you to implement your own updating function for that, but
it's quite likely you'll want to use the one in ``detan`` itself. The
``assignment_iteration`` function creates a closure over distances you provide.
The other name you import, ``AssignmentAnnealing``, is a class for the annealing
state.

Next, we need the pairwise dissimilarity matrix. Remember, this is symmetric, so
I just create a triangular matrix and add it to its own transpose:

.. code:: python

    distances = np.asarray((
        (0.0 , 2.1 , 0.10, 0.85, 0.2 , 0.78),
        (0.0 , 0.0 , 0.92, 0.05, 1.01, 0.01),
        (0.0 , 0.0 , 0.0 , 2.02, 0.15, 0.99),
        (0.0 , 0.0 , 0.0 , 0.0 , 1.30, 0.31),
        (0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 1.05),
        (0.0 , 0.0 , 0.0 , 0.0 , 0.0 , 0.0 ),
    ))

    distances = distances + distances.T

Each entry represents the "distance" between two things, so the diagonal has to
be all zeros (a thing has no distance from itself). Try to eyeball how the
clustering will go — thing #0 will probably be in the same group as thing #2
(distance of ``0.10``), but not thing #1 (distance of ``2.1``).

Let's say there are two groups:

.. code:: python

    groups = 2

The initial assignment expectatons should be randomised (really, each row must
contain distinct entries), and they need to sum to one:

.. code:: python

    initial_assignments = 0.5 + 0.1 * (np.random.random((6,groups)) - 0.5)
    row_sum = np.tile(initial_assignments.sum(1), (groups, 1)).T
    initial_assignments = initial_assignments/row_sum

An ``AssignmentAnnealing`` object is the state of the deterministic annealling
process, including the current temperature, current assignment expectations and
the last set of values from the last temperature step. Here we give it the
closure mentioned above, an initial set of random assignments, and a ratio of
``0.73`` to use for the temperature decrease.

.. code:: python

    annealer = AssignmentAnnealing(assignment_iteration(distances), initial_assignments, 0.73)

This is the loop where we actually do the annealing. An outer loop decreases the
temperature, and an inner loop does the fixed point iteration (the ``annealer``
object itself is an iterator that does this for you):

.. code:: python

    tolerance = 1e-6
    old_assignments = initial_assignments

    for _ in range(20):
        for new_assignments in annealer:
            if np.abs(new_assignments - old_assignments).max() < tolerance:
                break
            old_assignments = new_assignments
        annealer.cool()

More sophisticated calling code might try to account for the problems outlined
above (``NaN`` values in the expectation matrix, detecting convergence, etc.).
But the code above shows the fundamental structure of deterministic annealing.

Finally, the results.

.. code:: python

    print(annealer.assignments)

...gives us:

.. code:: python

    [[  3.29633866e-151   1.00000000e+000]
     [  1.00000000e+000   2.21285560e-174]
     [  1.18723351e-162   1.00000000e+000]
     [  1.00000000e+000   3.17951854e-163]
     [  1.80615908e-132   1.00000000e+000]
     [  1.00000000e+000   1.40074506e-107]]

Informally, the values seem to have converged to zero or one. (There's no
*objective* way to decide this, but for the demo, let's go with it.) So we could
just pick a completely arbitrary threshold and do this:

.. code:: python

    print(annealer.assignments > 1e-50)

...giving:

.. code:: python

    [[False  True]
     [ True False]
     [False  True]
     [ True False]
     [False  True]
     [ True False]]

This tells us that, as we expected, thing #0 and thing #2 are in the same group,
and in a different group to thing #1.
