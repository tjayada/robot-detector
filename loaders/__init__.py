from loaders import baxter, craves, dream, hydra, realsense_franka

# Each entry: (camera_info, iter_frames, count_frames).
LOADERS = {
    "baxter_real": (baxter.camera_info, baxter.iter_frames, baxter.count_frames),
    "panda_orb":   (dream.camera_info,  dream.iter_frames,  dream.count_frames),
    "craves":      (craves.camera_info, craves.iter_frames, craves.count_frames),
    # All three Hydra arms share the same measurement_N/ layout, so one loader.
    "hydra":       (hydra.camera_info,  hydra.iter_frames,  hydra.count_frames),
    "realsense_franka": (realsense_franka.camera_info, realsense_franka.iter_frames, realsense_franka.count_frames),
}
