import os
import shutil
import numpy as np
from trimesh.version import __version__ as trimesh_version
import trimesh as tm

def export_urdf(
        coacd_path, input_filename,
        output_directory,
        scale=1.0,
        color=[0.75, 0.75, 0.75],
        **kwargs):
    """
    Convert a Trimesh object into a URDF package for physics simulation.
    This breaks the mesh into convex pieces and writes them to the same
    directory as the .urdf file.

    Parameters
    ---------
    input_filename   : str
    output_directiry : str
                  The directory path for the URDF package

    Returns
    ---------
    mesh : Trimesh object
             Multi-body mesh containing convex decomposition
    """

    import lxml.etree as et
    # TODO: fix circular import
    from trimesh.exchange.export import export_mesh
    # Extract the save directory and the file name
    fullpath = os.path.abspath(output_directory)
    name = os.path.basename(fullpath)
    _, ext = os.path.splitext(name)

    if ext != '':
        raise ValueError('URDF path must be a directory!')

    # Create directory if needed
    if not os.path.exists(fullpath):
        os.mkdir(fullpath)
    elif not os.path.isdir(fullpath):
        raise ValueError('URDF path must be a directory!')

    # Perform a convex decomposition
    # if not exists:
    #     raise ValueError('No coacd available!')

    argstring = f' {input_filename} {os.path.join(output_directory, "decomposed.obj")}'

    # pass through extra arguments from the input dictionary
    for key, value in kwargs.items():
        argstring += ' -{} {}'.format(str(key),
                                      str(value))
    os.system(coacd_path + argstring + '\"')

    load_path  = os.path.join(output_directory, 'decomposed.obj')
    convex_pieces = list(tm.load(load_path, process=False).split())

    # Get the effective density of the mesh
    mesh = tm.load(input_filename, force="mesh", process=False)
    effective_density = mesh.volume / sum([
        m.volume for m in convex_pieces])

    # open an XML tree
    root = et.Element('robot', name='root')

    # Loop through all pieces, adding each as a link
    prev_link_name = None
    for i, piece in enumerate(convex_pieces):

        # Save each nearly convex mesh out to a file
        piece_name = '{}_convex_piece_{}'.format(name, i)
        piece_filename = '{}.obj'.format(piece_name)
        piece_filepath = os.path.join(fullpath, piece_filename)
        export_mesh(piece, piece_filepath)

        # Set the mass properties of the piece
        piece.center_mass = mesh.center_mass
        piece.density = effective_density * mesh.density

        link_name = 'link_{}'.format(piece_name)
        geom_name = '{}'.format(piece_filename)
        I = [['{:.2E}'.format(y) for y in x]  # NOQA
             for x in piece.moment_inertia]

        # Write the link out to the XML Tree
        link = et.SubElement(root, 'link', name=link_name)

        # Inertial information
        inertial = et.SubElement(link, 'inertial')
        et.SubElement(inertial, 'origin', xyz="0 0 0", rpy="0 0 0")
        # et.SubElement(inertial, 'mass', value='{:.2E}'.format(piece.mass))
        et.SubElement(
            inertial,
            'inertia',
            ixx=I[0][0],
            ixy=I[0][1],
            ixz=I[0][2],
            iyy=I[1][1],
            iyz=I[1][2],
            izz=I[2][2])
        # Visual Information
        visual = et.SubElement(link, 'visual')
        et.SubElement(visual, 'origin', xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(visual, 'geometry')
        et.SubElement(geometry, 'mesh', filename=geom_name,
                      scale="{:.4E} {:.4E} {:.4E}".format(scale,
                                                          scale,
                                                          scale))
        material = et.SubElement(visual, 'material', name='')
        et.SubElement(material,
                      'color',
                      rgba="{:.2E} {:.2E} {:.2E} 1".format(color[0],
                                                           color[1],
                                                           color[2]))

        # Collision Information
        collision = et.SubElement(link, 'collision')
        et.SubElement(collision, 'origin', xyz="0 0 0", rpy="0 0 0")
        geometry = et.SubElement(collision, 'geometry')
        et.SubElement(geometry, 'mesh', filename=geom_name,
                      scale="{:.4E} {:.4E} {:.4E}".format(scale,
                                                          scale,
                                                          scale))

        # Create rigid joint to previous link
        if prev_link_name is not None:
            joint_name = '{}_joint'.format(link_name)
            joint = et.SubElement(root,
                                  'joint',
                                  name=joint_name,
                                  type='fixed')
            et.SubElement(joint, 'origin', xyz="0 0 0", rpy="0 0 0")
            et.SubElement(joint, 'parent', link=prev_link_name)
            et.SubElement(joint, 'child', link=link_name)

        prev_link_name = link_name

    # Write URDF file
    tree = et.ElementTree(root)
    urdf_filename = '{}.urdf'.format(name)
    tree.write(os.path.join(fullpath, urdf_filename),
               pretty_print=True)

    # Write Gazebo config file
    root = et.Element('model')
    model = et.SubElement(root, 'name')
    model.text = name
    version = et.SubElement(root, 'version')
    version.text = '1.0'
    sdf = et.SubElement(root, 'sdf', version='1.4')
    sdf.text = '{}.urdf'.format(name)

    author = et.SubElement(root, 'author')
    et.SubElement(author, 'name').text = 'trimesh {}'.format(trimesh_version)
    et.SubElement(author, 'email').text = 'blank@blank.blank'

    description = et.SubElement(root, 'description')
    description.text = name

    tree = et.ElementTree(root)
    tree.write(os.path.join(fullpath, 'model.config'))

    return np.sum(convex_pieces)

 

coacd_path = '/bin/bash -c "source ~/anaconda3/etc/profile.d/conda.sh && conda activate adapose && coacd '
input_path = 'raw.obj'
output_path = './coacd'
coacd_params = {}
"""
    Here is the description of the parameters (sorted by importance).
        -i/--input: path for input mesh (.obj).
        -o/--output: path for output (.obj or .wrl).
        -t/--threshold: concavity threshold for terminating the decomposition (0.01~1), default = 0.05.
        -np/--no-prepocess: flag to disable manifold preprocessing, default = false. If your input is already manifold mesh, disabling the preprocessing can avoid introducing extra artifacts.
        -nm/--no-merge: flag to disable merge postprocessing, default = false.
        -mi/--mcts-iteration: number of search iterations in MCTS (60~2000), default = 100.
        -md/--mcts-depth: max search depth in MCTS (2~7), default = 3.
        -mn/--mcts-node: max number of child nodes in MCTS (10~40), default = 20.
        -pr/--prep-resolution: resolution for manifold preprocess (20~100), default = 50.
        -r/--resolution: sampling resolution for Hausdorff distance calculation (1e3~1e4), default = 2000.
        --pca: flag to enable PCA pre-processing, default = false.
        -k: value of k for Rv calculation, default = 0.3.
        --seed: random seed used for sampling, default = random().
"""

if os.path.exists('coacd'):
    shutil.rmtree('coacd')
os.makedirs('coacd')
export_urdf(
    coacd_path, 
    input_path, 
    output_path, 
    **coacd_params
)