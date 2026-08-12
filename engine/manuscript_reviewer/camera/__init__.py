"""Camera-motion subsystem (Phase 4).

Estimates GLOBAL background/image motion per shot — never inferring camera
movement from foreground subject movement alone. Estimation is deterministic
(FFT phase correlation, RNG-free); classification asserts on derived motion
class and direction, never on raw float magnitudes. 2D global motion never
claims dolly/track; scale is reported as SCALE_CHANGE, not "dolly".
"""
