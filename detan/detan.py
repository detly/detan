# Copyright 2016 Jason Heeris <jason.heeris@gmail.com>
#
# This library is licensed under the 3-clause BSD license distributed with this
# software.

import numpy as np

def assignment_potential(assignments, distances):
    """
    This calculates the potentials (the 'ε*[i,λ]') from the assignment
    expectations matrix for deterministic annealing.

    The assignment matrix must contain entries in (0, 1) (note open endpoints),
    and the distance matrix must be symmetric and contain only zeros on the
    diagonal.

    :param assignments: assignment expectations
    :param distances: pairwise distance matrix
    :return: assignment potentials
    """
    n, k = assignments.shape
    mdm = (assignments.T).dot(distances).dot(assignments)
    sum_M1 = np.outer(np.ones((n,)), assignments.sum(0)) - assignments

    sum_outer = (
          distances.dot(assignments)
        - (1 / (2 * sum_M1)) * np.outer(np.ones((n,)), np.diag(mdm))
    )

    potential = (1 / (sum_M1 + 1)) * sum_outer
    return potential


def assignment_expectations(potentials, T):
    """
    Calculates assignment expectations (the 〈M[i,λ]〉) from assignment potentials
    for deterministic annealing.

    :param potentials: the assignment potentials (any real numbers)
    :param T: the Lagrangian parameter (temperature) for deterministic annealing
      (strictly greater than zero)
    """
    exp_potential = np.exp(-potentials/T)
    potentials    = exp_potential / np.outer(exp_potential.sum(1), np.ones((1, potentials.shape[1])))
    return potentials


def assignment_iteration(distances):
    """
    Composition of :func:`assignment_potential` and
    :func:`assignment_expectations` suitable for fixed point iteration. This
    will raise an exception if the assignment matrix contains any NaN entries.
    """
    def closure(assignments, T):
        potentials = assignment_potential(assignments, distances)
        new_assignments = assignment_expectations(potentials, T)

        if np.isnan(new_assignments).any():
            raise ValueError("NaN in computed assignment expectations")

        return new_assignments

    return closure


class AssignmentAnnealing:
    """
    The deterministic annealing algorithm starts with randomised assignment
    expectations (the 〈M[i,λ]〉). Fixed point iteration (usually involving an
    intermediate potential calculation) at a particular "temperature" should
    cause the assignment expectations to converge. Lowering the temperature will
    eventually cause them to converge closer to {0, 1}.

    Instances of this class hold the state of convergence and perform the
    iterations. It can be used as an iterator, like so::

        # Random initial assignments
        M = 1 - np.random.random((n ,k))

        # Normalise the assignments so each row sum is 1
        M = M/np.tile(M.sum(1), (k, 1)).T

        # Create deterministic annealing state
        annealer = AssignmentAnnealing(iterator_function, M, 0.73)

        # Tolerance for convergence
        tolerance = 1e-6

        for temperature_steps in range(20):
            for new_assignments in annealer:
                if (new_assignments - old_assignments).abs().max() < tolerance:
                    break
            annealer.cool()

        print(np.round(annealer.assignments).astype(int))

    This is a very simple example; there are modes of failure you might need to
    account for depending on your inputs. Since the iterator returned by the
    annealer is just itself, you can use the built-in function :func:`next()`
    instead::

        for temperature_steps in range(20):
            for iterations in range(20):
                next(annealer)
            annealer.cool()

    The :meth:`cool()` method will lower the
    temperature by the ratio given in the constructor.

    The annealing object also keeps track of the last set of assignment
    potentials and temperature when :meth:`cool()` was called.

    Due to the maths involved combined with floating point imprecision, it is
    possible that the fixed point iteration sometimes results in NaN entries in
    the assignment matrix. The :meth:`reheat()` method will restore the
    remembered temperature and assignments, and the caller can then eg. change
    the ratio, or take some other action to continue annealing.

    The `ratio` member is a part of the API and can be changed at any time. It
    must always be strictly between 0 and 1. It has the same meaning as the
    `ratio` parameter in the constructor.

    The `assignments` member is part of the API, but should not be assigned to,
    only read from.
    """

    def __init__(self, function, initial_assignments, temperature_ratio):
        """
        Create a new deterministic annealing state. The given function should
        have the interface::

            def function(assignments, temperature):
                # ...fixed point iteration function...
                return new_assignments

        The `assignments` are the assignment expectation values that you are
        aiming to have converge to {0, 1}. The sum of any row in this matrix
        should be 1. The assignment matrix returned by the function should have
        the same shape and constraints as the input.

        The temperature is the Lagrangian parameter used for annealing. It is
        strictly positive, and should get lower as the annealing progresses
        (although it may increase temporarily if convergence proves difficult).

        The initial temperature will always be 1.

        :param function: a function suitable for fixed-point iteration,
          described above
        :param initial_assignments: the initial values of the assignment
          expectations (usually a random assignment)
        :param temperature_ratio: a number strictly between 0 and 1; at each
          step, the temperature will be lowered by this factor
        """
        self.function = function
        self.assignments = initial_assignments
        self.temperature = 1
        self.ratio = temperature_ratio
        self._stash()


    def _stash(self):
        # Stores temperature and assignments in case annealing produces NaN
        # values and the caller wants to back up a step.
        self._stashed = (self.temperature, self.assignments)


    def __iter__(self):
        return self


    def __next__(self):
        """
        When the annealer is used as an iterator (either in a `for` loop or by
        calling :func:`next(annealer)`, the value produced is the next set of
        assignment potentials from the fixed point iteration.
        """
        next_assignments = self.function(self.assignments, self.temperature)
        self.assignments = next_assignments
        return next_assignments


    def cool(self):
        """
        Reduces the temperature used for further function calls by the
        user-supplied ratio.
        """
        assert (self.ratio > 0.0 and self.ratio < 1.0), "Ratio is not in (0, 1)"
        self._stash()
        self.temperature *= self.ratio


    def reheat(self):
        """
        Restores the result and temperature from before the :meth:`cool()`
        method was called.
        """
        self.temperature, self.assignments = self._stashed
