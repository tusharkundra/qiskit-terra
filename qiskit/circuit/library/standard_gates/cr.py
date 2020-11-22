# This code is part of Qiskit.
#
# (C) Copyright IBM 2017.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""CRGate."""

from qiskit.circuit.gate import Gate
from qiskit.qasm import pi


class CRGate(Gate):
    """CRGate."""
    def __init__(self, theta, label=None):
        super().__init__("crgate%s" % theta, 2, [theta])

    def inverse(self):
        r"""Return inverted CR gate (itself)."""
        return CRGate(-self.params[0])

    def _define(self):
        """
        gate crgate(theta) a {crgate(theta) a;}
        """
        super()._define()
