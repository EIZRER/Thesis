# Photogrammetry Pipeline

Desktop GUI for photogrammetry reconstruction using **COLMAP** (sparse) + **OpenMVS** (dense/mesh/texture).

## Pipeline

```
Images → COLMAP → Sparse Model → InterfaceCOLMAP → OpenMVS Scene
       → DensifyPointCloud → ReconstructMesh → RefineMesh → TextureMesh
       → Dense Point Cloud + Mesh + Textured OBJ
```

## Prerequisites

1. **COLMAP** — https://colmap.github.io/install.html
2. **OpenMVS** — https://github.com/cdcseacave/openMVS/wiki/Building

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

Edit `config/config.yaml` and update the paths to your COLMAP and OpenMVS executables:

```yaml
executables:
  colmap: "C:/Program Files/COLMAP/COLMAP.bat"
  interface_colmap: "C:/OpenMVS/bin/InterfaceCOLMAP.exe"
  densify: "C:/OpenMVS/bin/DensifyPointCloud.exe"
  reconstruct_mesh: "C:/OpenMVS/bin/ReconstructMesh.exe"
  refine_mesh: "C:/OpenMVS/bin/RefineMesh.exe"
  texture_mesh: "C:/OpenMVS/bin/TextureMesh.exe"
```

## Run

```bash
python main.py
```

## Output

After a successful run, outputs are collected in `workspace/output/`:

| File | Description |
|------|-------------|
| `dense_point_cloud.ply` | Dense 3D point cloud |
| `mesh.ply` | Raw mesh |
| `textured_mesh.obj` | Final textured mesh |
| `textured_mesh.mtl` | Material file |

## Notes

- GPU support requires COLMAP and OpenMVS compiled with CUDA.
- Use **absolute paths** for image folders on Windows.
- The `Refine Mesh` and `Texture Mesh` steps can be disabled in the GUI.
