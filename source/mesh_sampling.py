import math
import heapq
import numpy as np
import scipy.sparse as sp
from chumpy.utils import row, col
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from source.shape_model_utils import overwrite_vtkpoly


def get_vert_connectivity(poly_data=None, verts=None, faces=None):
    """Returns a sparse matrix (of size #verts x #verts) where each nonzero
    element indicates a neighborhood relation. For example, if there is a
    nonzero element in position (15,12), that means vertex 15 is connected
    by an edge to vertex 12.

    Either `poly_data` or `verts` and `faces` must be given as argument.
    """

    if poly_data is not None:
        verts = _get_poly_vertices(poly_data)
        faces = _get_poly_faces(poly_data)
    vpv = sp.csc_matrix((len(verts), len(verts)))

    # for each column in the faces...
    for i in range(3):
        IS = faces[:, i]
        JS = faces[:, (i+1) % 3]
        data = np.ones(len(IS))
        ij = np.vstack((row(IS.flatten()), row(JS.flatten())))
        mtx = sp.csc_matrix((data, ij), shape=vpv.shape)
        vpv = vpv + mtx + mtx.T

    return vpv


def get_vertices_per_edge(poly_data=None, verts=None, faces=None):
    """Returns an Ex2 array of adjacencies between vertices, where
    each element in the array is a vertex index. Each edge is included
    only once. If output of get_faces_per_edge is provided, this is used to
    avoid call to get_vert_connectivity()

    Either `poly_data` or `verts` and `faces` must be given as argument.
    """

    if poly_data is not None:
        verts = _get_poly_vertices(poly_data)
        faces = _get_poly_faces(poly_data)
    vc = sp.coo_matrix(get_vert_connectivity(verts=verts, faces=faces))
    result = np.hstack((col(vc.row), col(vc.col)))
    result = result[result[:, 0] < result[:, 1]]  # for uniqueness

    return result


def vertex_quadrics(poly_data=None, verts=None, faces=None):
    """Computes a quadric for each vertex in the poly.

    Either `poly_data` or `verts` and `faces` must be given as argument.

    Returns:
       v_quadrics: an (N x 4 x 4) array, where N is # vertices.
    """

    # Allocate quadrics
    if poly_data is not None:
        verts = _get_poly_vertices(poly_data)
        faces = _get_poly_faces(poly_data)
    v_quadrics = np.zeros((len(verts), 4, 4,))

    # For each face...
    for f_idx in range(len(faces)):

        # Compute normalized plane equation for that face
        vert_idxs = faces[f_idx]
        vertices = np.hstack((verts[vert_idxs],
                              np.array([1, 1, 1]).reshape(-1, 1)))
        u, s, v = np.linalg.svd(vertices)
        eq = v[-1, :].reshape(-1, 1)
        eq = eq / (np.linalg.norm(eq[0:3]))

        # Add the outer product of the plane equation to the
        # quadrics of the vertices for this face
        for k in range(3):
            v_quadrics[faces[f_idx, k], :, :] += np.outer(eq, eq)

    return v_quadrics


def setup_deformation_transfer(source_poly, target_poly, use_normals=False):
    target_v = _get_poly_vertices(target_poly)
    source_v = _get_poly_vertices(source_poly)
    source_f = _get_poly_faces(source_poly)

    rows = np.zeros(3 * target_v.shape[0])
    cols = np.zeros(3 * target_v.shape[0])
    coeffs_v = np.zeros(3 * target_v.shape[0])
    coeffs_n = np.zeros(3 * target_v.shape[0])

    source_mesh = convert_vtk_to_mesh(source_poly)
    nearest_faces, nearest_parts, nearest_vertices = \
        source_mesh.compute_aabb_tree().nearest(target_v, True)
    nearest_faces = nearest_faces.ravel().astype(np.int64)
    nearest_parts = nearest_parts.ravel().astype(np.int64)
    nearest_vertices = nearest_vertices.ravel()

    for i in range(target_v.shape[0]):
        # Closest triangle index
        f_id = nearest_faces[i]
        # Closest triangle vertex ids
        nearest_f = source_f[f_id]

        # Closest surface point
        nearest_v = nearest_vertices[3 * i:3 * i + 3]
        # Distance vector to the closest surface point
        dist_vec = target_v[i] - nearest_v

        rows[3 * i:3 * i + 3] = i * np.ones(3)
        cols[3 * i:3 * i + 3] = nearest_f

        n_id = nearest_parts[i]
        if n_id == 0:
            # Closest surface point in triangle
            A = np.vstack((source_v[nearest_f])).T
            coeffs_v[3 * i:3 * i + 3] = np.linalg.lstsq(A, nearest_v, rcond=None)[0]
        elif n_id > 0 and n_id <= 3:
            # Closest surface point on edge
            A = np.vstack((source_v[nearest_f[n_id - 1]],
                           source_v[nearest_f[n_id % 3]])).T
            tmp_coeffs = np.linalg.lstsq(A, target_v[i], rcond=None)[0]
            coeffs_v[3 * i + n_id - 1] = tmp_coeffs[0]
            coeffs_v[3 * i + n_id % 3] = tmp_coeffs[1]
        else:
            # Closest surface point a vertex
            coeffs_v[3 * i + n_id - 4] = 1.0

    #    if use_normals:
    #        A = np.vstack((vn[nearest_f])).T
    #        coeffs_n[3 * i:3 * i + 3] = np.linalg.lstsq(A, dist_vec, rcond=None)[0]

    # coeffs = np.hstack((coeffs_v, coeffs_n))
    # rows = np.hstack((rows, rows))
    # cols = np.hstack((cols, source.v.shape[0] + cols))
    matrix = sp.csc_matrix((coeffs_v, (rows, cols)),
                           shape=(target_v.shape[0], source_v.shape[0]))
    return matrix


def qslim_decimator_transformer(poly_data, factor=None, n_verts_desired=None):
    """Return a simplified version of this `poly_data`.

    A Qslim-style approach is used here.

    :param factor: fraction of the original vertices to retain
    :param n_verts_desired: number of the original vertices to retain
    :returns: new_faces: An Fx3 array of faces, mtx: Transformation matrix
    """

    orig_verts = _get_poly_vertices(poly_data)
    orig_faces = _get_poly_faces(poly_data)
    if factor is None and n_verts_desired is None:
        raise Exception('Need either factor or n_verts_desired.')

    if n_verts_desired is None:
        n_verts_desired = math.ceil(len(orig_verts) * factor)

    Qv = vertex_quadrics(verts=orig_verts, faces=orig_faces)

    # fill out a sparse matrix indicating vertex-vertex adjacency
    # from psbody.mesh.topology.connectivity import get_vertices_per_edge
    vert_adj = get_vertices_per_edge(verts=orig_verts, faces=orig_faces)
    # vert_adj = sp.lil_matrix((len(mesh.v), len(mesh.v)))
    # for f_idx in range(len(mesh.f)):
    #     vert_adj[mesh.f[f_idx], mesh.f[f_idx]] = 1

    vert_adj = sp.csc_matrix(
        (vert_adj[:, 0] * 0 + 1, (vert_adj[:, 0], vert_adj[:, 1])),
        shape=(len(orig_verts), len(orig_verts)))

    vert_adj = vert_adj + vert_adj.T
    vert_adj = vert_adj.tocoo()

    def collapse_cost(Qv, r, c, v):
        Qsum = Qv[r, :, :] + Qv[c, :, :]
        p1 = np.vstack((v[r].reshape(-1, 1), np.array([1]).reshape(-1, 1)))
        p2 = np.vstack((v[c].reshape(-1, 1), np.array([1]).reshape(-1, 1)))

        destroy_c_cost = p1.T.dot(Qsum).dot(p1)
        destroy_r_cost = p2.T.dot(Qsum).dot(p2)
        result = {
            'destroy_c_cost': destroy_c_cost,
            'destroy_r_cost': destroy_r_cost,
            'collapse_cost': min([destroy_c_cost, destroy_r_cost]),
            'Qsum': Qsum}
        return result

    # construct a queue of edges with costs
    queue = []
    for k in range(vert_adj.nnz):
        r = vert_adj.row[k]
        c = vert_adj.col[k]

        if r > c:
            continue

        cost = collapse_cost(Qv, r, c, orig_verts)['collapse_cost']
        heapq.heappush(queue, (cost, (r, c)))

    # decimate
    collapse_list = []
    nverts_total = len(orig_verts)
    faces = orig_faces.copy()
    while nverts_total > n_verts_desired:
        e = heapq.heappop(queue)
        r = e[1][0]
        c = e[1][1]
        if r == c:
            continue

        cost = collapse_cost(Qv, r, c, orig_verts)
        if cost['collapse_cost'] > e[0]:
            heapq.heappush(queue, (cost['collapse_cost'], e[1]))
            continue
        else:

            # update old vert idxs to new one,
            # in queue and in face list
            if cost['destroy_c_cost'] < cost['destroy_r_cost']:
                to_destroy = c
                to_keep = r
            else:
                to_destroy = r
                to_keep = c

            collapse_list.append([to_keep, to_destroy])

            #  replace "to_destroy" vertidx with "to_keep" vertidx
            np.place(faces, faces == to_destroy, to_keep)

            # same for queue
            which1 = [idx for idx in range(len(queue))
                      if queue[idx][1][0] == to_destroy]
            which2 = [idx for idx in range(len(queue))
                      if queue[idx][1][1] == to_destroy]
            for k in which1:
                queue[k] = (queue[k][0], (to_keep, queue[k][1][1]))
            for k in which2:
                queue[k] = (queue[k][0], (queue[k][1][0], to_keep))

            Qv[r, :, :] = cost['Qsum']
            Qv[c, :, :] = cost['Qsum']

            a = faces[:, 0] == faces[:, 1]
            b = faces[:, 1] == faces[:, 2]
            c = faces[:, 2] == faces[:, 0]

            # remove degenerate faces
            def logical_or3(x, y, z):
                return np.logical_or(x, np.logical_or(y, z))

            faces_to_keep = np.logical_not(logical_or3(a, b, c))
            faces = faces[faces_to_keep, :].copy()

        nverts_total = (len(np.unique(faces.flatten())))

    new_faces, mtx = _get_sparse_transform(faces, len(orig_verts))
    return new_faces, mtx


def _get_sparse_transform(faces, num_original_verts):
    verts_left = np.unique(faces.flatten())
    IS = np.arange(len(verts_left))
    JS = verts_left
    data = np.ones(len(JS))

    mp = np.arange(0, np.max(faces.flatten()) + 1)
    mp[JS] = IS
    new_faces = mp[faces.copy().flatten()].reshape((-1, 3))

    ij = np.vstack((IS.flatten(), JS.flatten()))
    mtx = sp.csc_matrix((data, ij),
                        shape=(len(verts_left), num_original_verts))

    return new_faces, mtx


# generate and return down and upsampling matrices (D and U) using qslim decimation
# use them to generate sequence of downsampled polys (list M)
# return adjacency matrices (list A) of those polys as well
def generate_transform_matrices(poly_data, factors):
    """Generates len(factors) polys, each of them is scaled by factors[i] and
       computes the transformations between them.

    Returns:
       M: a set of poly data downsampled from `poly_data` by a factor
       specified in `factors`.
       A: Adjacency matrix for each of the polys
       D: Downsampling transforms between each of the polys
       U: Upsampling transforms between each of the polys
    """

    factors = map(lambda x: 1.0 / x, factors)
    M, A, D, U = [], [], [], []
    # append to list A the adjacency matrix of the input poly "poly_data"
    A.append(get_vert_connectivity(poly_data=poly_data))
    # append to list M the input poly "poly_data"
    M.append(poly_data)

    for factor in factors:
        # Mesh decimation can reduce mesh complexity, and keep geometry unchanged as much as possible.
        # Here we use the qslim mesh decimator to downsample the mesh M[-1] (the last downsampled mesh)
        # by a factor "factor"
        # we get back the downsampling factor ds_f and the downsampling matrix ds_D to downsample the last poly in M
        ds_f, ds_D = qslim_decimator_transformer(M[-1], factor=factor)
        # append the output downsampling matrix to list of ds matrices
        D.append(ds_D)

        # apply downsampling matrix to last poly P in M to get the downsampled version ds_P
        verts = _get_poly_vertices(M[-1])
        # append ds_P to the list of polys M
        new_verts = ds_D.dot(verts)
        new_poly = _make_new_poly(vertices=new_verts, faces=ds_f)
        M.append(new_poly)
        # append the adjacency matrix of ds_P to list A
        A.append(get_vert_connectivity(verts=new_verts, faces=ds_f))
        # get the upsampling matrix to get ds_P to P, append this upsampling matrix to list U 
        U.append(setup_deformation_transfer(M[-1], M[-2]))

    return M, A, D, U


def convert_vtk_to_mesh(poly_data):
    """Converts a vtk polydata mesh into a ply file and returns the ply
    mesh"""
    from psbody.mesh import Mesh

    verts = _get_poly_vertices(poly_data)
    faces = _get_poly_faces(poly_data)

    ply_mesh = Mesh(v=verts, f=faces)
    return ply_mesh


def _get_poly_vertices(poly_data):
    return vtk_to_numpy(poly_data.GetPoints().GetData())


def _get_poly_faces(poly_data):
    n_poly = poly_data.GetNumberOfPolys()
    faces = vtk_to_numpy(poly_data.GetPolys().GetData())
    faces = faces.reshape(n_poly, faces[0] + 1)
    assert (faces[:, 0] == 3).all(), "not triangle faces or different format"
    return faces[:, 1:]


def _make_new_poly(vertices, faces):
    poly_data = vtk.vtkPolyData()
    faces = np.hstack((np.ones(shape=(faces.shape[0], 1), dtype=np.int) * 3,
                       faces))
    new_poly = overwrite_vtkpoly(poly_data, points=vertices, polys=faces)
    return new_poly
