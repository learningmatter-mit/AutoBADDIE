import torch
import materialbuilder

# from materialbuilder import utils
from materialbuilder.graphbuilder import Graph, GraphBatch

# from mendeleev import element as Element
from rdkit.Chem import AllChem as Chem

# these dictionaries are from mendeleev:
# from mendeleev import element as Element
# {z: Element(z).mass for z in torch.arange(1,100).tolist()}
Z_TO_ATOMIC_MASSES = {
    1: 1.008,
    2: 4.002602,
    3: 6.94,
    4: 9.0121831,
    5: 10.81,
    6: 12.011,
    7: 14.007,
    8: 15.999,
    9: 18.998403163,
    10: 20.1797,
    11: 22.98976928,
    12: 24.305,
    13: 26.9815385,
    14: 28.085,
    15: 30.973761998,
    16: 32.06,
    17: 35.45,
    18: 39.948,
    19: 39.0983,
    20: 40.078,
    21: 44.955908,
    22: 47.867,
    23: 50.9415,
    24: 51.9961,
    25: 54.938044,
    26: 55.845,
    27: 58.933194,
    28: 58.6934,
    29: 63.546,
    30: 65.38,
    31: 69.723,
    32: 72.63,
    33: 74.921595,
    34: 78.971,
    35: 79.904,
    36: 83.798,
    37: 85.4678,
    38: 87.62,
    39: 88.90584,
    40: 91.224,
    41: 92.90637,
    42: 95.95,
    43: 97.90721,
    44: 101.07,
    45: 102.9055,
    46: 106.42,
    47: 107.8682,
    48: 112.414,
    49: 114.818,
    50: 118.71,
    51: 121.76,
    52: 127.6,
    53: 126.90447,
    54: 131.293,
    55: 132.90545196,
    56: 137.327,
    57: 138.90547,
    58: 140.116,
    59: 140.90766,
    60: 144.242,
    61: 144.91276,
    62: 150.36,
    63: 151.964,
    64: 157.25,
    65: 158.92535,
    66: 162.5,
    67: 164.93033,
    68: 167.259,
    69: 168.93422,
    70: 173.045,
    71: 174.9668,
    72: 178.49,
    73: 180.94788,
    74: 183.84,
    75: 186.207,
    76: 190.23,
    77: 192.217,
    78: 195.084,
    79: 196.966569,
    80: 200.592,
    81: 204.38,
    82: 207.2,
    83: 208.9804,
    84: 209.0,
    85: 210.0,
    86: 222.0,
    87: 223.0,
    88: 226.0,
    89: 227.0,
    90: 232.0377,
    91: 231.03588,
    92: 238.02891,
    93: 237.0,
    94: 244.0,
    95: 243.0,
    96: 247.0,
    97: 247.0,
    98: 251.0,
    99: 252.0,
}
SYMBOLS_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
}

ATOMIC_MASSES_TO_Z = {v: k for k, v in Z_TO_ATOMIC_MASSES.items()}
Z_TO_SYMBOLS = {v: k for k, v in SYMBOLS_TO_Z.items()}
SYMBOLS_TO_ATOMIC_MASSES = {
    Z_TO_SYMBOLS[z]: Z_TO_ATOMIC_MASSES[z] for z in Z_TO_ATOMIC_MASSES.keys()
}


class Molecule(Graph):
    def __init__(self, atomic_nums, parent=None, A=None):
        super(Molecule, self).__init__(parent, A)
        self.AddNodeLabel("atomic_num", atomic_nums, torch.long)
        atomic_masses = [
            Z_TO_ATOMIC_MASSES[z] for z in self.atomic_nums.view(-1).tolist()
        ]
        self.AddNodeLabel("mass", atomic_masses, torch.float)

    def assign_types(self, types, type_name):
        self.AddNodeLabel(type_name, types, torch.long)

    @property
    def atomic_nums(self):
        return self._data["node"]["atomic_num"]


class Geometry(Molecule):
    def __init__(self, atomic_nums, xyz, A=None, parent=None):
        super(Geometry, self).__init__(atomic_nums, parent, A)
        xyz = self._clean_tensor(xyz, dtype=torch.float)
        if xyz.shape[1] == 4:
            xyz = xyz[:, -3:]
        self.AddNodeLabel("xyz", xyz)

    @property
    def xyz(self):
        return self._data["node"]["xyz"]

    def to_xyz(self, filename):
        xyz = torch.cat([self.atomic_nums.to(torch.float), self.xyz], dim=1).tolist()
        file = open(filename, "w")
        file.write(str(self.N) + "\n")
        file.write("Atoms. Timestep: 0" + "\n")
        for row in xyz:
            file.write(
                "{} {} {} {}\n".format(
                    Element(int(row[0])).symbol, str(row[1]), str(row[2]), str(row[3])
                )
            )
        file.close()


def molecule_from_smiles(smiles, return_adjmat=False):
    rdkit_mol = Chem.MolFromSmiles(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    atomic_nums = [int(atom.GetAtomicNum()) for atom in rdkit_mol.GetAtoms()]
    A = Chem.GetAdjacencyMatrix(rdkit_mol)
    molecule = Molecule(atomic_nums, A=A)
    if return_adjmat:
        return (molecule, A)
    else:
        return molecule


def geometry_from_smiles(smiles, dim=2, return_adjmat=False, return_rdkit_mol=False):
    rdkit_mol = Chem.MolFromSmiles(smiles)
    print(smiles)
    rdkit_mol = Chem.AddHs(rdkit_mol)
    atomic_nums = [int(atom.GetAtomicNum()) for atom in rdkit_mol.GetAtoms()]
    A = Chem.GetAdjacencyMatrix(rdkit_mol)
    if dim != 0:
        if dim == 2:
            Chem.Compute2DCoords(rdkit_mol)
        elif dim == 3:
            for attempt in range(100):
                conformerID = Chem.EmbedMolecule(rdkit_mol, Chem.ETKDG())
                if conformerID == 0:
                    break
            Chem.UFFOptimizeMolecule(rdkit_mol)
        rdkit_conformer = rdkit_mol.GetConformer(0)
        xyz = rdkit_conformer.GetPositions()
    else:
        xyz = torch.zeros(len(A), 3).to(torch.float)
    geometry = Geometry(atomic_nums, xyz, A=A)
    to_return = []
    to_return.append(geometry)
    if return_adjmat:
        to_return.append(A)
    if return_rdkit_mol:
        to_return.append(rdkit_mol)
    if len(to_return) == 1:
        return to_return[0]
    return tuple(to_return)
