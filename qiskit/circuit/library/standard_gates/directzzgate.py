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

"""ZZInteraction"""

from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.qasm import pi


class DirectZZGate(Gate):
    """ZZInteraction(theta) gate accomplished at the pulse level."""
    def __init__(self, theta, label=None):
        super().__init__("directzzgate%s" % theta, 2, [theta], label=label)

    def _define(self):
        # super()._define()
        """ """
        # pylint: disable=cyclic-import
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u2 import U2Gate
        from .directrxgate import DirectRXGate
        from .cr import CRGate

        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [
            (DirectRXGate(pi), [q[0]], []),
            (U2Gate(0, pi), [q[1]], []),
            (CRGate(-self.params[0]), [q[0], q[1]], []),
            (U2Gate(0, pi), [q[1]], []),
        ]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)

        self.definition = qc


