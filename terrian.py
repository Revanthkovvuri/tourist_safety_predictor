import numpy as np

print("Recalculating step-by-step physical fatigue...")

# ── 1. Effort Multiplier ──────────────────────────────────────────────────────
conditions_speed = [
    (df['speed_kmh'] <= 2),                                 # Stationary / Standing
    (df['speed_kmh'] > 2)  & (df['speed_kmh'] <= 5),       # Walking
    (df['speed_kmh'] > 5)  & (df['speed_kmh'] <= 10),      # Jogging
    (df['speed_kmh'] > 10) & (df['speed_kmh'] <= 15),      # Running
    (df['speed_kmh'] > 15) & (df['speed_kmh'] <= 40),      # Cycling (moderate effort)
    (df['speed_kmh'] > 40)                                  # Driving / Transit
]
choices_effort = [0.0, 1.0, 1.5, 2.0, 0.3, 0.05]
df['effort_multiplier'] = np.select(conditions_speed, choices_effort, default=1.0)

# ── 2. Terrain Multiplier ─────────────────────────────────────────────────────
terrain_multipliers = {
    'Highland':  1.40,   # was 'Extreme Mountain'
    'Hilly':     1.20,   # slightly reduced — 1.15 was too close to Hilly
    'Rural':     1.05,   # flat open land, minor extra effort
    'Flatland':  1.00,   # neutral baseline
    'Urban':     1.00,
    'Coastal':   0.95,   # flat, slight tailwind assumed
    'Trail':     1.10,   # uneven surface adds effort
    'Highway':   0.85,   # smooth surface, less resistance
    'Unknown':   1.00
}
df['terrain_multiplier'] = df['terrain'].map(terrain_multipliers).fillna(1.0)  

# ── 3. Step Fatigue ───────────────────────────────────────────────────────────
df['step_fatigue'] = df['dist_delta_km'] * df['effort_multiplier'] * df['terrain_multiplier']

# ── 4. Grade Penalty ──────────────────────────────────────────────────────────

raw_grade_penalty = np.where(
    df['speed_kmh'] <= 40,
    df['grade'] / 10,
    0
)
grade_penalty = np.clip(raw_grade_penalty, -0.5, 1.0)  # downhill recovery capped, uphill capped
df['step_fatigue'] = df['step_fatigue'] * (1 + grade_penalty)

# ── 5. Rest Recovery ──────────────────────────────────────────────────────────

rest_recovery = np.where(df['speed_kmh'] <= 2, -0.002, 0)  # small per-step reduction
df['step_fatigue'] = df['step_fatigue'] + rest_recovery

# ── 6. Cumulative Fatigue per Day ─────────────────────────────────────────────
df['fatigue_index'] = df.groupby('date')['step_fatigue'].cumsum()

# ── 7. Scale and Cap ──────────────────────────────────────────────────────────
SCALE_FACTOR = 5.0  # Tune this to shift your 0–100 baseline
df['fatigue_index'] = np.clip(df['fatigue_index'] * SCALE_FACTOR, 0, 100)  # also floor at 0

# ── 8. Cleanup ────────────────────────────────────────────────────────────────
df = df.drop(columns=['effort_multiplier', 'terrain_multiplier', 'step_fatigue'])

print("Done. Fatigue index recalculated.\n")
print(df[['speed_kmh', 'grade', 'terrain', 'fatigue_index']].head(20))