import torch, itertools, copy
from scipy.linalg import block_diag
import numpy as np
from collections import OrderedDict


# this epsilon will be used by torch.clamp is added because the gradient of torch.acos(1) or torch.acos(-1) will be inf and mess up the training!!
# issue documented here - https://github.com/pytorch/pytorch/issues/8069
EPSILON_FOR_LINEAR_ANGLE = 1e-7


############################### SHARED GEOMETRY FUNCTIONS #############################################
def safe_acos(cos, epsilon=1e-7):
    angle = torch.acos(torch.clamp(cos, -1 + epsilon, 1 - epsilon))
    return angle


def get_b(xyz, bonds):
    b = (xyz[bonds[:, 1]] - xyz[bonds[:, 0]]).pow(2).sum(-1).pow(0.5)
    return b


def get_theta(xyz, angles, nan_flag=False, nan_inx=None):
    # clamp is added because the gradient of torch.acos(1) or torch.acos(-1) will be inf and mess up the training!!
    # issue documented here - https://github.com/pytorch/pytorch/issues/8069
    if nan_flag:
        print("index in trouble:", nan_inx)
        import ipdb

        ipdb.set_trace()
    angle_vec1 = xyz[angles[:, 0]] - xyz[angles[:, 1]]
    angle_vec2 = xyz[angles[:, 2]] - xyz[angles[:, 1]]

    norm = (angle_vec1.pow(2).sum(1) * angle_vec2.pow(2).sum(1)).sqrt()
    cos_theta = torch.clamp(
        (angle_vec1 * angle_vec2).sum(1) / norm,
        -1 + EPSILON_FOR_LINEAR_ANGLE,
        1 - EPSILON_FOR_LINEAR_ANGLE,
    )
    theta = torch.acos(cos_theta)
    return (theta, cos_theta)


def get_phi(xyz, dihedrals):
    # calculates the dihedral angles based on the xyz coordinates of a batch as well as the indices of each atom in each dihedral
    # the dihedral angle is calculated by first finding the norms of the two planes that make up a dihedral (cross1, cross2),
    # then finding the angle between those two plane norms
    # The v and S variables are used to express all dihedral angles between -180 and 180 deg (equivalently: -pi and pi rad).
    vec1 = xyz[dihedrals[:, 0]] - xyz[dihedrals[:, 1]]
    vec2 = xyz[dihedrals[:, 2]] - xyz[dihedrals[:, 1]]
    vec3 = xyz[dihedrals[:, 1]] - xyz[dihedrals[:, 2]]
    vec4 = xyz[dihedrals[:, 3]] - xyz[dihedrals[:, 2]]
    cross1 = torch.cross(vec1, vec2, dim=-1)
    cross2 = torch.cross(vec3, vec4, dim=-1)
    norm = (cross1.pow(2).sum(1) * cross2.pow(2).sum(1)).sqrt()
    cos_phi = torch.clamp(
        (cross1 * cross2).sum(1) / norm,
        -1 + EPSILON_FOR_LINEAR_ANGLE,
        1 - EPSILON_FOR_LINEAR_ANGLE,
    )

    v = vec2 / (vec2.pow(2).sum(1).pow(0.5).unsqueeze(1))
    PHI1 = vec1 - (vec1 * v).sum(1).unsqueeze(1) * v
    PHI2 = vec4 - (vec4 * v).sum(1).unsqueeze(1) * v
    S = torch.sign((torch.cross(PHI1, PHI2, dim=-1) * v).sum(1))
    phi = np.pi + S * (torch.acos(cos_phi) - np.pi)
    return (phi, cos_phi)


def get_chi(xyz, impropers):
    A, B, C, D = 0, 1, 2, 3
    permutations = [[A, B, C, D], [C, B, D, A], [D, B, A, C]]
    chi = []
    for permutation in permutations:
        A, B, C, D = permutation
        rBA = xyz[impropers[:, B]] - xyz[impropers[:, A]]
        rBC = xyz[impropers[:, B]] - xyz[impropers[:, C]]
        rBD = xyz[impropers[:, B]] - xyz[impropers[:, D]]
        RBA = rBA.pow(2).sum(1).pow(0.5).view(-1, 1)
        RBC = rBC.pow(2).sum(1).pow(0.5).view(-1, 1)
        RBD = rBD.pow(2).sum(1).pow(0.5).view(-1, 1)
        uBA = rBA / RBA
        uBC = rBC / RBC
        uBD = rBD / RBD
        sinCBD = torch.cross(uBC, uBD, dim=-1).pow(2).sum(1).pow(0.5).view(-1, 1)
        sinchiABCD = (torch.cross(uBC, uBD, dim=-1) * uBA).sum(1).view(-1, 1) / sinCBD
        chiABCD = torch.asin(sinchiABCD)
        chi.append(chiABCD)
    chi = torch.cat(chi, dim=1)
    chi = chi.mean(1)
    return chi


def get_d(xyz, impropers):
    A, B, C, D = 0, 1, 2, 3
    rAB = xyz[impropers[:, A]] - xyz[impropers[:, B]]
    rAC = xyz[impropers[:, A]] - xyz[impropers[:, C]]
    rAD = xyz[impropers[:, A]] - xyz[impropers[:, D]]

    rACD = torch.cross(rAC, rAD, dim=-1)
    uACD = rACD / (rACD.pow(2).sum(1).pow(0.5).view(-1, 1))
    d = (rAB * uACD).sum(1).view(-1, 1)
    d = d * uACD
    return d


############################### SHARED DERIVATIVE FUNCTIONS ###########################################


def get_db(xyz, bonds):
    b = get_b(xyz, bonds)
    device = xyz.device
    r = xyz[bonds]
    I = torch.eye(2).to(device)
    b = b.view(-1, 1).unsqueeze(1)
    rb = r[:, [1]] - r[:, [0]]
    db = rb * (I[1] - I[0]).view(1, 2, 1) / b
    return db


def get_dtheta(xyz, angles):
    theta, cos_theta = get_theta(xyz, angles)
    device = angles.device
    r = xyz[angles]
    r0 = r[:, [0]] - r[:, [1]]
    R0 = r0.pow(2).sum(2).pow(0.5).unsqueeze(1)
    r2 = r[:, [2]] - r[:, [1]]
    R2 = r2.pow(2).sum(2).pow(0.5).unsqueeze(1)
    cos_theta = cos_theta.view(-1, 1).unsqueeze(1)
    I = torch.eye(3).to(device)
    A = (r0 / (R0 * R2) - cos_theta * r2 / (R2.pow(2))) * (I[2] - I[1]).view(1, 3, 1)
    B = (r2 / (R0 * R2) - cos_theta * r0 / (R0.pow(2))) * (I[0] - I[1]).view(1, 3, 1)
    dtheta = -((1 - cos_theta.pow(2)).pow(-0.5)) * (A + B)
    # if ~np.all(dtheta.isfinite().tolist()):
    #     print(dtheta.min(dim=0))
    #     theta2, cos_theta2 = get_theta(xyz, angles, nan_flag=True, nan_inx=dtheta.min(dim=0)[1][0][0].item())
    #     import ipdb
    #     ipdb.set_trace()
    return dtheta


def get_dphi(xyz, dihedrals):
    phi, cos_phi = get_phi(xyz, dihedrals)
    device = dihedrals.device
    N = len(dihedrals)
    r = xyz[dihedrals].unsqueeze(2)
    e = torch.eye(3).view(1, 1, 3, 3).expand(N, 4, 3, 3).to(device)
    I = torch.eye(4).to(device)
    cos_phi = torch.cos(phi.view(-1, 1)).unsqueeze(2)
    r12 = (r[:, [1]] - r[:, [0]]).expand(N, 4, 3, 3)
    r23 = (r[:, [2]] - r[:, [1]]).expand(N, 4, 3, 3)
    r123 = torch.cross(r12, r23, dim=3)
    R123 = r123.pow(2).sum(3).pow(0.5).unsqueeze(3)
    delta23 = (I[2] - I[1]).view(1, 4, 1, 1).expand(N, 4, 3, 3)
    delta12 = (I[1] - I[0]).view(1, 4, 1, 1).expand(N, 4, 3, 3)
    dr123 = torch.cross(r12, e, dim=3) * delta23 - torch.cross(r23, e, dim=3) * delta12
    dr123 = dr123 * -1
    r23 = (r[:, [2]] - r[:, [1]]).expand(N, 4, 3, 3)
    r34 = (r[:, [3]] - r[:, [2]]).expand(N, 4, 3, 3)
    r234 = torch.cross(r23, r34, dim=3)
    R234 = r234.pow(2).sum(3).pow(0.5).unsqueeze(3)
    delta34 = (I[3] - I[2]).view(1, 4, 1, 1).expand(N, 4, 3, 3)
    delta23 = (I[2] - I[1]).view(1, 4, 1, 1).expand(N, 4, 3, 3)
    dr234 = torch.cross(r23, e, dim=3) * delta34 - torch.cross(r34, e, dim=3) * delta23
    dr234 = dr234 * -1
    dcos_phi = 0.0
    dcos_phi = dcos_phi + (r123 * dr234).sum(3)
    dcos_phi = dcos_phi + (r234 * dr123).sum(3)
    dcos_phi = dcos_phi / (R123 * R234).squeeze(3)
    dcos_phi = dcos_phi - cos_phi * ((r234 / R234.pow(2)) * dr234).sum(3)
    dcos_phi = dcos_phi - cos_phi * ((r123 / R123.pow(2)) * dr123).sum(3)
    dcos_phi = dcos_phi * -1
    dphi = -(1 - cos_phi.pow(2)).pow(-0.5) * dcos_phi
    return dphi


def get_dchi(xyz, impropers):
    displacements = xyz.unsqueeze(1).expand(len(xyz), len(xyz), 3)
    displacements = displacements.transpose(0, 1) - displacements
    A, B, C, D = 0, 1, 2, 3
    permutations = [[A, B, C, D], [C, B, D, A], [D, B, A, C]]
    dchi = []
    for permutation in permutations:
        A, B, C, D = permutation
        rBA = displacements[impropers[:, A], impropers[:, B]]
        rBC = displacements[impropers[:, C], impropers[:, B]]
        rBD = displacements[impropers[:, D], impropers[:, B]]
        RBA = rBA.pow(2).sum(1).pow(0.5).view(-1, 1)
        RBC = rBC.pow(2).sum(1).pow(0.5).view(-1, 1)
        RBD = rBD.pow(2).sum(1).pow(0.5).view(-1, 1)
        uBA = rBA / RBA
        uBC = rBC / RBC
        uBD = rBD / RBD
        sinCBD = torch.cross(uBC, uBD, dim=-1).pow(2).sum(1).pow(0.5).view(-1, 1)
        sinchiABCD = (torch.cross(uBC, uBD, dim=-1) * uBA).sum(1).view(-1, 1) / sinCBD
        chiABCD = torch.asin(sinchiABCD)
        a = (uBA * torch.cross(uBC, uBD, dim=-1)).sum(1).view(-1, 1, 1)
        r = xyz[impropers[:, [A, B, C, D]]]
        I = torch.eye(4).to(impropers.device)
        dcosCBD = 0.0
        dcosCBD = dcosCBD + ((r[:, [2]] - r[:, [1]]) * (I[3] - I[1]).view(1, 4, 1)) / (
            RBC * RBD
        ).unsqueeze(1)
        dcosCBD = dcosCBD + ((r[:, [3]] - r[:, [1]]) * (I[2] - I[1]).view(1, 4, 1)) / (
            RBC * RBD
        ).unsqueeze(1)
        cosCBD = (uBC * uBD).sum(1).view(-1, 1, 1)
        dcosCBD = dcosCBD - cosCBD * (
            (r[:, [3]] - r[:, [1]]) * (I[3] - I[1]).view(1, 4, 1)
        ) / (RBD.pow(2)).unsqueeze(1)
        dcosCBD = dcosCBD - cosCBD * (
            (r[:, [2]] - r[:, [1]]) * (I[2] - I[1]).view(1, 4, 1)
        ) / (RBC.pow(2)).unsqueeze(1)
        dsinCBD = cosCBD * (-(1 - cosCBD.pow(2)).pow(-0.5)) * dcosCBD
        b = -(1 / sinCBD.unsqueeze(1).pow(2)) * dsinCBD
        c = sinCBD.unsqueeze(1).pow(-1)
        d = 0.0
        RBA = RBA.unsqueeze(1)
        RBC = RBC.unsqueeze(1)
        RBD = RBD.unsqueeze(1)
        N = RBA * RBC * RBD
        dN = 0.0
        dN = (
            dN
            + RBC * RBD * ((r[:, [0]] - r[:, [1]]) * (I[0] - I[1]).view(1, 4, 1)) / RBA
        )
        dN = (
            dN
            + RBA * RBD * ((r[:, [2]] - r[:, [1]]) * (I[2] - I[1]).view(1, 4, 1)) / RBC
        )
        dN = (
            dN
            + RBA * RBC * ((r[:, [3]] - r[:, [1]]) * (I[3] - I[1]).view(1, 4, 1)) / RBD
        )
        e = 0.0
        e = e + torch.cross(rBA, rBC, dim=-1).unsqueeze(1) * (I[3] - I[1]).view(1, 4, 1)
        e = e + torch.cross(rBD, rBA, dim=-1).unsqueeze(1) * (I[2] - I[1]).view(1, 4, 1)
        e = e + torch.cross(rBC, rBD, dim=-1).unsqueeze(1) * (I[0] - I[1]).view(1, 4, 1)
        d = (
            d
            + (rBA * torch.cross(rBC, rBD, dim=-1)).sum(1).view(-1, 1, 1)
            * (-1 / N.pow(2))
            * dN
        )
        d = d + (1 / N) * e
        dsinchi = a * b + c * d
        dchi.append((1 - sinchiABCD.unsqueeze(1).pow(2)).pow(-0.5) * dsinchi)
    dchi = torch.stack(dchi).mean(0)
    dchi = dchi * -1
    return dchi


############################### GEOMETRIES ############################################################


def get_bond_geometry(xyz, bonds):
    b = get_b(xyz, bonds)
    return b


def get_angle_geometry(xyz, angles):
    theta, cos_theta = get_theta(xyz, angles)
    bonds = angles[:, [1, 0, 1, 2]].view(-1, 2)
    b = get_b(xyz, bonds).view(-1, 2)
    return (cos_theta, theta, b)


def get_dihedral_geometry(xyz, dihedrals):
    phi, cos_phi = get_phi(xyz, dihedrals)
    angles = torch.stack([dihedrals[:, :3], dihedrals[:, -3:]], dim=1).view(-1, 3)
    theta, cos_theta = get_theta(xyz, angles)
    cos_theta = cos_theta.view(-1, 2)
    theta = theta.view(-1, 2)
    bonds = dihedrals[:, [0, 1, 1, 2, 2, 3]].view(-1, 2)
    b = get_b(xyz, bonds).view(-1, 3)
    return (cos_phi, phi, theta, b)


def get_improper_geometry(xyz, impropers):
    chi = get_chi(xyz, impropers)
    d = get_d(xyz, impropers)
    i, j, k, l = 0, 1, 2, 3
    angles = torch.stack(
        [impropers[:, [i, j, k]], impropers[:, [i, j, l]], impropers[:, [k, j, l]]],
        dim=1,
    ).view(-1, 3)
    theta, cos_theta = get_theta(xyz, angles)
    cos_theta = cos_theta.view(-1, 3)
    theta = theta.view(-1, 3)
    return (chi, cos_theta, theta, d)


############################### DERIVATIVES ###########################################################


def get_bond_derivatives(xyz, bonds):
    db = get_db(xyz, bonds)
    return db


def get_angle_derivatives(xyz, angles):
    dtheta = get_dtheta(xyz, angles)
    bonds = angles[:, [0, 1, 1, 2]].view(-1, 2)
    _db = get_db(xyz, bonds).view(-1, 2, 2, 3)
    db = torch.zeros(len(_db), 2, 3, 3).to(xyz.device)
    db[:, 0, [0, 1]] = db[:, 0, [0, 1]] + _db[:, 0]
    db[:, 1, [1, 2]] = db[:, 1, [1, 2]] + _db[:, 1]
    return (dtheta, db)


def get_dihedral_derivatives(xyz, dihedrals):
    import time

    start = time.time()
    dphi = get_dphi(xyz, dihedrals)
    print("\t\t\t\ttime to get dphi[s]", time.time() - start)
    start = time.time()
    angles = torch.stack([dihedrals[:, :3], dihedrals[:, -3:]], dim=1).view(-1, 3)
    _dtheta = get_dtheta(xyz, angles).view(-1, 2, 3, 3)
    print("\t\t\t\ttime to get dtheta command[s]", time.time() - start)
    start = time.time()
    dtheta = torch.zeros(len(_dtheta), 2, 4, 3).to(xyz.device)
    dtheta[:, 0, [0, 1, 2]] = dtheta[:, 0, [0, 1, 2]] + _dtheta[:, 0]
    dtheta[:, 1, [1, 2, 3]] = dtheta[:, 1, [1, 2, 3]] + _dtheta[:, 1]
    print("\t\t\t\ttime to get reorder dthetas[s]", time.time() - start)
    start = time.time()
    bonds = dihedrals[:, [0, 1, 1, 2, 2, 3]].view(-1, 2)
    _db = get_db(xyz, bonds).view(-1, 3, 2, 3)
    db = torch.zeros(len(_db), 3, 4, 3).to(xyz.device)
    db[:, 0, [0, 1]] = db[:, 0, [0, 1]] + _db[:, 0]
    db[:, 1, [1, 2]] = db[:, 1, [1, 2]] + _db[:, 1]
    db[:, 2, [2, 3]] = db[:, 2, [2, 3]] + _db[:, 2]
    print("\t\t\t\ttime to get dbs[s]", time.time() - start)
    start = time.time()
    return (dphi, dtheta, db)


def get_improper_derivatives(xyz, impropers):
    dchi = get_dchi(xyz, impropers)
    i, j, k, l = 0, 1, 2, 3
    angles = torch.stack(
        [impropers[:, [i, j, k]], impropers[:, [i, j, l]], impropers[:, [k, j, l]]],
        dim=1,
    ).view(-1, 3)
    _dtheta = get_dtheta(xyz, angles).view(-1, 3, 3, 3)
    dtheta = torch.zeros(len(_dtheta), 3, 4, 3).to(xyz.device)
    dtheta[:, 0, [0, 1, 2]] = dtheta[:, 0, [0, 1, 2]] + _dtheta[:, 0]
    dtheta[:, 1, [0, 1, 3]] = dtheta[:, 1, [0, 1, 3]] + _dtheta[:, 1]
    dtheta[:, 2, [2, 1, 3]] = dtheta[:, 2, [2, 1, 3]] + _dtheta[:, 2]
    return (dchi, dtheta)


############################### INDICES ###############################################################


def index_of(inp, source, max_index):
    # inp = list of atoms in cluster as their atomic num
    # source = list of unique atomic nums in inp, in increasing order
    # max_index = highest atomic number in source + 1

    X = torch.randint(
        0, 9999999999, (max_index,), device=source.device
    )  # creates list with length highest-atomic-number-in-smiles+1 of random ints
    inp = X[
        inp
    ].sum(
        1
    )  # creates a list of length #atoms in cluster but with random ints for each element instead of atomic num
    try:
        source = X[
            source
        ].sum(
            1
        )  # list of length number-of-unique-elements containing the random int from above in place where atomic num used to be
    except:
        import pdb

        pdb.set_trace()
    try:
        source, sorted_index, inverse = np.unique(
            source.tolist(), return_index=True, return_inverse=True
        )
    except:
        import pdb

        pdb.set_trace()
    try:
        index = torch.cat([torch.tensor(source, device=inp.device), inp]).unique(
            sorted=True, return_inverse=True
        )[1][-len(inp) :]
    except:
        import pdb

        pdb.set_trace()
    try:
        index = torch.tensor(sorted_index, device=index.device)[index]
    except:
        import pdb

        pdb.set_trace()
    return index


def get_bonds_in_angles(graph, angles, device):
    bonds = angles[:, [0, 1, 1, 2]].view(-1, 2).to(device)
    ref_bonds = graph.get_data("bond", "index").to(device)
    try:
        bonds = index_of(bonds, source=ref_bonds, max_index=graph.N).view(-1, 2)
    except:
        import pdb

        pdb.set_trace()
    return bonds


def get_bonds_in_dihedrals(graph, dihedrals, device):
    bonds = dihedrals[:, [0, 1, 1, 2, 2, 3]].view(-1, 2).to(device)
    ref_bonds = graph.get_data("bond", "index").to(device)
    bonds = index_of(bonds, source=ref_bonds, max_index=graph.N).view(-1, 3)
    return bonds


def get_angles_in_dihedrals(graph, dihedrals, device):
    angles = dihedrals[:, [0, 1, 2, 1, 2, 3]].view(-1, 3).to(device)
    ref_angles = graph.get_data("angle", "index").to(device)
    angles = index_of(angles, source=ref_angles, max_index=graph.N).view(-1, 2)
    return angles


def get_bonds_in_impropers(graph, impropers, device):
    bonds = impropers[:, [1, 0, 1, 2, 1, 3]].view(-1, 3, 2).to(device)
    ref_bonds = graph.get_data("bond", "index").to(device)
    bonds = index_of(bonds.view(-1, 2), source=ref_bonds, max_index=graph.N).view(-1, 3)
    return bonds


def get_angles_in_impropers(graph, impropers, device):
    angles = (
        torch.stack(
            [impropers[:, [0, 1, 2]], impropers[:, [0, 1, 3]], impropers[:, [2, 1, 3]]],
            dim=1,
        )
        .view(-1, 3)
        .to(device)
    )
    ref_angles = graph.get_data("angle", "index").to(device)
    angles = index_of(angles, source=ref_angles, max_index=graph.N).view(-1, 3)
    return angles
