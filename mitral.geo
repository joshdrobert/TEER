Merge "segmented_valve_mesh_smoothed.stl";

// Wrap the closed surface into a surface loop and a volume
Surface Loop(1) = {1};
Volume(1) = {1};

// Give that volume a physical tag
Physical Volume("fluid", 1) = {1};
