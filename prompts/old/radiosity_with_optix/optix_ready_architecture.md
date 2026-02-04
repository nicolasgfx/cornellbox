# OptiX-Ready Data Structure Architecture

## Overview

The data structures have been redesigned for direct OptiX compatibility, eliminating the need for complex transformations between CPU and GPU representations.

## Key Design Decisions

### 1. **Indexed Mesh Representation**

**Old approach (Week 2):**
```cpp
struct Triangle {
    Vector3 v0, v1, v2;  // Embedded vertices
};
vector<Triangle> triangles;  // Duplicated vertices
```

**New approach (OptiX-ready):**
```cpp
struct IndexedMesh {
    vector<Vector3> vertices;     // Shared vertex buffer
    vector<uint32_t> indices;     // Triangle indices (groups of 3)
    vector<uint32_t> patchIds;    // Per-triangle patch mapping
};
```

**Benefits:**
- No vertex duplication
- Direct OptiX upload (no transformation)
- Less memory usage
- Matches GPU buffer layout exactly

---

### 2. **Patch Structure**

**Purpose:** Store radiosity state and material properties

```cpp
struct Patch {
    // Geometry (precomputed)
    Vector3 center, normal;
    float area;
    
    // Material
    Vector3 emission, reflectance;
    
    // Radiosity state
    Vector3 B;         // Current radiosity
    Vector3 B_unshot;  // Unshot radiosity
    
    // Mesh mapping
    int firstTriangleIndex;
    int triangleCount;
};
```

---

### 3. **Scene Structure**

**Purpose:** Central data structure for both CPU radiosity and GPU rendering

```cpp
class RadiosityScene {
    vector<Patch> patches;      // Radiosity domain
    IndexedMesh mesh;           // Geometry domain (OptiX-compatible)
};
```

---

## Data Flow

### Scene Construction
```
Authoring → Create Patch → Add Geometry → Update Mapping
```

Example:
```cpp
scene.addQuadPatch(
    corner0, corner1, corner2, corner3,
    material,
    subdivisionU, subdivisionV
);
```

This:
1. Creates one Patch with computed properties
2. Subdivides quad into indexed triangles
3. Associates triangles with patch ID
4. Updates patch triangle count

---

### OptiX Upload (Week 3)
```
CPU → GPU
```

```cpp
// Vertex buffer
optixBufferSetData(vertices, mesh.getVertexDataPtr(), mesh.getVertexDataSize());

// Index buffer
optixBufferSetData(indices, mesh.getIndexDataPtr(), mesh.getIndexDataSize());

// Patch IDs (for ray hit → patch lookup)
optixBufferSetData(patchIds, mesh.patchIds.data(), ...);
```

**No transformation needed!**

---

### Ray Tracing → Radiosity Mapping
```
OptiX Hit → Triangle Index → Patch ID → Patch Data
```

```cpp
// In OptiX hit program:
uint32_t triangleIndex = optixGetPrimitiveIndex();
uint32_t patchId = patchIds[triangleIndex];

// Back on CPU after ray query:
Patch& patch = scene.patches[patchId];
```

---

## Memory Layout Comparison

### Old (Week 2)
```
Triangle 0: {v0, v1, v2, normal}  // 4 * Vector3 = 48 bytes
Triangle 1: {v0, v2, v3, normal}  // 4 * Vector3 = 48 bytes
...
Total for 1000 tris: ~48 KB (with duplication)
```

### New (OptiX-ready)
```
Vertices: [v0, v1, v2, v3, ...]    // n * 12 bytes
Indices: [0,1,2, 0,2,3, ...]       // 1000*3 * 4 bytes = 12 KB
PatchIds: [0, 0, 1, 1, ...]        // 1000 * 4 bytes = 4 KB
Total for 1000 tris: ~22 KB (shared vertices)
```

**~50% memory savings** for typical meshes

---

## OptiX Integration Points

### 1. **Geometry Upload**
```cpp
// Build OptiX acceleration structure
OptixBuildInput buildInput = {};
buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
buildInput.triangleArray.vertexBuffers = &vertexBuffer;
buildInput.triangleArray.numVertices = mesh.vertexCount();
buildInput.triangleArray.indexBuffer = indexBuffer;
buildInput.triangleArray.numIndexTriplets = mesh.triangleCount();
```

### 2. **Visibility Testing**
```cpp
// Cast ray from patch i to patch j
bool visible(uint32_t patchI, uint32_t patchJ) {
    Vector3 origin = patches[patchI].center;
    Vector3 target = patches[patchJ].center;
    
    // OptiX ray trace
    // Hit program checks: if hitPatchId != patchJ, occluded
}
```

### 3. **Final Rendering**
```cpp
// For each pixel:
//   Trace camera ray
//   Get triangle hit
//   Lookup patch ID
//   Return patch.B (radiosity)
```

---

## Benefits Summary

### ✅ **OptiX Compatibility**
- No CPU→GPU transformation
- Direct buffer upload
- Matches OptiX memory layout

### ✅ **Memory Efficiency**
- ~50% less memory
- Shared vertices
- Compact indices

### ✅ **Clear Abstractions**
- Patch = radiosity entity
- Triangle = geometry entity
- Scene = container

### ✅ **Fast Lookups**
- O(1) triangle → patch
- O(1) patch → triangles
- Contiguous memory

### ✅ **Simplicity**
- Single mesh representation
- No duplicate data
- No synchronization needed

---

## Migration from Week 2

**Old code (Triangle-based):**
```cpp
Mesh mesh;
mesh.addQuad(v0, v1, v2, v3);
for (auto& tri : mesh.triangles) {
    // Process triangle
}
```

**New code (Indexed):**
```cpp
RadiosityScene scene;
uint32_t patchId = scene.addQuadPatch(v0, v1, v2, v3, material, 8, 8);
// Geometry is automatically indexed and patch-mapped
```

---

## Next Steps (Week 3)

1. **Update CornellBox** to use `RadiosityScene`
2. **Implement OptiX context** setup
3. **Upload mesh** to OptiX
4. **Implement visibility testing** via shadow rays
5. **Implement radiosity solver** using patch structure
6. **Render final image** via OptiX path tracing

All with minimal data transformation!
