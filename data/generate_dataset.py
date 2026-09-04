"""
Generates a realistic synthetic Student Placement dataset.
(Kaggle.com isn't reachable from this build sandbox, so this script creates a
dataset with the SAME structure/columns as the popular Kaggle "Campus
Placement" datasets, with realistic correlations baked in, so the ML model
trained on it behaves sensibly. You can drop in a real Kaggle CSV later —
just keep the same column names — and re-run train_model.py.)
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 800

tenth = np.clip(np.random.normal(75, 10, N), 40, 100).round(2)
twelfth = np.clip(tenth + np.random.normal(0, 8, N), 40, 100).round(2)
cgpa = np.clip((tenth + twelfth) / 20 + np.random.normal(0, 0.8, N), 5.0, 10.0).round(2)
internships = np.random.poisson(1.1, N).clip(0, 5)
projects = np.random.poisson(2.0, N).clip(0, 6)
backlogs = np.random.poisson(0.4, N).clip(0, 5)
communication_skill = np.clip(np.random.normal(6.5, 1.8, N), 1, 10).round(0)
extra_curricular = np.random.binomial(1, 0.45, N)
degree = np.random.choice(["B.Tech", "B.Sc", "BCA", "M.Tech", "MCA"], N, p=[0.45, 0.15, 0.2, 0.1, 0.1])
interest = np.random.choice(
    ["Web Development", "Data Science", "Core/Mechanical", "Machine Learning", "Networking", "Testing"], N
)

# Score that drives placement probability
score = (
    0.05 * tenth + 0.05 * twelfth + 1.4 * cgpa
    + 1.1 * internships + 0.55 * projects
    - 1.6 * backlogs + 0.7 * communication_skill
    + 0.6 * extra_curricular
)
score = (score - score.mean()) / score.std() * 3.2
prob = 1 / (1 + np.exp(-score))
placed = np.random.binomial(1, prob)

package_lpa = np.where(
    placed == 1,
    np.clip(3 + cgpa * 0.6 + internships * 0.5 + np.random.normal(0, 1.0, N), 2.5, 30),
    0,
).round(2)

df = pd.DataFrame({
    "tenth_percent": tenth,
    "twelfth_percent": twelfth,
    "degree": degree,
    "cgpa": cgpa,
    "internships": internships,
    "projects": projects,
    "backlogs": backlogs,
    "communication_skill": communication_skill.astype(int),
    "extra_curricular": extra_curricular,
    "interest": interest,
    "placed": placed,
    "package_lpa": package_lpa,
})

df.to_csv("/home/claude/placement_predictor/data/student_placement_dataset.csv", index=False)
print(df.shape)
print(df["placed"].value_counts(normalize=True))
print(df.head())
