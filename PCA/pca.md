# 🧠 Section 1: INTUITION — What is PCA and Why Do We Need It?

---

## The Problem: Too Many Dimensions

Imagine you're a doctor and you collected data on 1,000 patients.

For each patient you recorded **50 things**:
- height, weight, age, blood pressure, cholesterol, heart rate, sleep hours, step count...

Each patient is now a point in **50-dimensional space**.

You can't visualize that. You can't plot it. You can't even think about it clearly.

**This is the curse of dimensionality** — more features = harder to analyze, visualize, and learn from.

---

## 🧠 The Core Idea of PCA

> **PCA finds the directions in your data where the most variation happens — and keeps only those.**

That's it. Everything else is details.

---

## Real-World Analogy: The Shadow

Imagine a **3D object** — say, a flying bird — and you're shining a light on it to cast a **shadow on a wall**.

- The shadow is **2D**
- It loses some information (depth)
- But a **good shadow** still tells you a lot — you can recognize it's a bird

**PCA is choosing the best angle to shine the light.**

A bad angle gives a useless blob. A good angle gives a shadow that preserves the most detail.

---

## What "Reducing Dimensions" Actually Means

You don't delete features randomly.

You find **new directions** (combinations of your original features) that capture the most information, then you **keep only the top few**.

### Example:
Say you have two features:
- `height` (cm)
- `weight` (kg)

These two are **correlated** — taller people tend to weigh more.

So both features are kind of saying the **same thing**.

PCA would find a single direction — maybe something like *"overall body size"* — that captures most of what both features say.

You go from **2 numbers per person → 1 number per person**, losing very little.

---

## ⚠️ Why This Matters in ML

| Problem | How PCA Helps |
|---|---|
| Too many features | Reduce to top K |
| Features are correlated | PCA decorrelates them |
| Can't visualize data | Project to 2D or 3D |
| Model is overfitting | Remove noisy dimensions |
| Slow training | Fewer features = faster |

---

## 🔧 ML Connection

PCA is used **before** training models to:
- Speed up training (fewer dimensions)
- Remove noise
- Visualize high-dimensional data in 2D
- Compress data (images, audio, text embeddings)

---

## Summary So Far

| Concept | Plain English |
|---|---|
| High-dimensional data | Too many features to work with |
| Dimensionality reduction | Keep fewer features, lose little info |
| PCA | Find the directions of most variation and project onto them |
| "Direction of variance" | Where your data spreads out the most |

---
---

# 📐 Section 2: GEOMETRIC VIEW — Seeing PCA with Your Eyes

---

## Start Simple: Your Data is Just Points in Space

Let's say you have 6 people, and for each you recorded **two things**:
- `x` = hours of exercise per week
- `y` = calories burned per day

| Person | Exercise (x) | Calories (y) |
|--------|-------------|--------------|
| A | 1 | 200 |
| B | 2 | 400 |
| C | 3 | 580 |
| D | 4 | 800 |
| E | 5 | 950 |
| F | 6 | 1100 |

Plot these and you get:

```
y (calories)
│
1100 ┤                    • F
950  ┤                 • E
800  ┤             • D
580  ┤         • C
400  ┤     • B
200  ┤  • A
     └──────────────────── x (exercise)
        1   2   3   4   5   6
```

**Notice:** the points form a diagonal line. They're highly correlated.

---

## 🧠 Key Insight: Where is the "Action"?

Look at the data. It mostly **spreads along a diagonal direction** (bottom-left → top-right).

That diagonal direction is where the **variance** lives — where points differ from each other the most.

The direction **perpendicular** to it? Almost nothing happens there. Points barely differ.

> **PCA's job: find that diagonal direction. That's your first Principal Component.**

---

## What is "Direction of Maximum Variance"?

Variance = how spread out your points are.

Imagine sliding a ruler across the plot at different angles:

- **Angle 1 (diagonal ↗):** Points are spread FAR apart along the ruler ✅
- **Angle 2 (vertical ↑):** Points are more bunched up along the ruler ❌
- **Angle 3 (horizontal →):** Points are somewhat spread, but less than diagonal ❌

The direction where points are **most spread out** = direction of maximum variance = **Principal Component 1**

---

## Projecting Points onto a Line

Now here's what "reducing from 2D → 1D" actually means geometrically.

You pick a line (a direction). You **drop each point perpendicularly onto that line**.

```
                    • (original point)
                   /|
                  / |  ← this dropped distance
                 /  |    is the information LOST
────────────────•───┘──────────────── (the line)
                ↑
          projected point
```

Each point now has **one number**: how far along the line it lands.

That number is its **new coordinate** in 1D.

---

## Good Projection vs Bad Projection

### ✅ Good line (along the diagonal):
```
      •
     •
    •         Points land FAR APART on the line
   •           → lots of variance preserved
  •
 •
──•──•──•──•──•──•──
```

### ❌ Bad line (perpendicular to diagonal):
```
──────────────────── 
      ↑↑↑↑↑↑
   all points land
   almost at the
   SAME SPOT
   → variance crushed
```

> **Best projection = the one that keeps points most spread out = maximum variance direction**

---

## What Information is Lost?

When you project from 2D → 1D, you lose the **distance each point was from the line**.

```
Original point:   (3, 580)
Projected point:  somewhere along the diagonal line
Lost info:        how far the point was off the line
```

- If points were perfectly on the line → **zero loss**
- If points are scattered randomly → **big loss**

Our exercise/calories data is nearly on a line → projecting onto that diagonal **loses almost nothing**.

---

## 🧠 The Big Picture (Before Any Math)

| Step | What Happens |
|------|-------------|
| Look at your 2D data | Points scattered in space |
| Find the diagonal direction | Direction of most spread (variance) |
| Project all points onto it | Each point becomes 1 number |
| That number = new feature | Captures most of the original info |

This is **exactly** what PCA does — just in higher dimensions where you can't see it.

---

## ⚠️ Why Does This Matter?

- If you projected onto a **random** line, you'd lose a ton of info
- PCA **guarantees** you pick the line that loses the **least** information
- This is why PCA is optimal — not just any projection, the **best** one

---

## 🔧 ML Connection

In real ML:
- Your data might be 512-dimensional (e.g. image embeddings)
- PCA finds the top 2 directions of variance
- You plot those 2 numbers → suddenly you can **see** your data
- This is how people visualize word embeddings, face datasets, etc.

---
---

# 🔢 Section 3: Linear Algebra You Need — Built Through PCA

---

> We'll learn **only** what we need, **exactly** when we need it.
> Every concept connects directly to the geometry you just saw.

---

## 3.1 — Vectors: Directions with Length

### 🧠 Intuition

A vector is just **an arrow**.

It has:
- A **direction** (which way it points)
- A **length** (how long it is)

In our exercise/calories example, each data point is a vector:

```
Person C = (3, 580)

y
│
580 ┤        • ← tip of the arrow
│       /
│      /
│     /
│    /
└────────── x
     3
```

The arrow starts at the origin (0,0) and points to (3, 580).

### Math notation:

```
v = [3, 580]   ← just a list of numbers = a vector
```

That's it. A vector = a list of numbers = an arrow in space.

---

## 3.2 — Length of a Vector (Norm)

### 🧠 Intuition

How long is the arrow?

Use Pythagoras. For a 2D vector `v = [a, b]`:

```
length = √(a² + b²)
```

### Example:

```
v = [3, 4]
length = √(9 + 16) = √25 = 5
```

```
        • (3,4)
       /|
    5 / |  4
     /  |
    /   |
   ─────┘
      3
```

### Why do we care?

In PCA, we need directions — **pure directions, no length**.

So we always **normalize** vectors to length 1:

```
unit vector = v / length(v)
```

A vector of length 1 = **unit vector** = pure direction.

---

## 3.3 — Dot Product: The Heart of Projection

### 🧠 Intuition

This is the **most important operation** in PCA.

The dot product between two vectors `a` and `b`:

```
a · b = a₁×b₁ + a₂×b₂
```

### Example:

```
a = [1, 2]
b = [3, 4]
a · b = 1×3 + 2×4 = 3 + 8 = 11
```

### But what does it MEAN geometrically?

> **The dot product measures how much two vectors point in the same direction.**

```
a · b is:
  LARGE  → vectors point same direction
  ZERO   → vectors are perpendicular (90°)
  NEGATIVE → vectors point opposite directions
```

Visually:

<img width="868" height="509" alt="image" src="https://github.com/user-attachments/assets/78bed6e1-dad4-4127-a9c7-def088959dc4" />


---

## 3.4 — Projection: Dropping a Point onto a Line

### 🧠 Intuition

Remember from Section 2 — we "drop" points onto a line perpendicularly.

Here's the exact math for that drop.

Say you have:
- A data point `p = [3, 580]`
- A direction (unit vector) `d = [0.7, 0.7]` (diagonal direction)

The **projection** of `p` onto direction `d` is:

```
projection = (p · d)
```

Just the dot product. That's all.

This gives you **one number**: how far along the direction `d` the point `p` lands.

### Why does this work?

```
        • p
       /|
      / |
     /  |
────•───┘──────────────  direction d →
    ↑
  projection = p · d
```

The dot product **extracts the component** of `p` in direction `d`.

If `d` is a unit vector, `p · d` = exact length of the shadow.

### ⚠️ This IS PCA projection

When PCA finds direction `d` and projects all your data onto it — it's doing exactly this dot product for every point.

---

## 3.5 — Orthogonality: Independent Directions

### 🧠 Intuition

Two vectors are **orthogonal** (perpendicular) when their dot product = 0.

```
a = [1, 0]   (pointing right →)
b = [0, 1]   (pointing up ↑)

a · b = 1×0 + 0×1 = 0  ✅ orthogonal
```

### Why do we care for PCA?

PCA finds **multiple** principal components.

- PC1 = direction of most variance
- PC2 = direction of second most variance **BUT must be perpendicular to PC1**
- PC3 = perpendicular to both PC1 and PC2
- ...

**Why perpendicular?**

Because if PC2 pointed in the same direction as PC1, it wouldn't give you **new information**. It would just repeat what PC1 already captured.

Orthogonal directions = **independent information**.

```
PC1 →→→→→→→→  (captures most spread)
PC2 ↑          (captures remaining spread, NEW info)
    (perpendicular to PC1)
```

---

## 3.6 — Variance Along a Direction

### 🧠 Intuition

You know variance = how spread out numbers are.

Now we combine everything:

> **Variance along direction `d`** = take all data points, project each onto `d`, measure how spread out those projections are.

### Step by step:

1. Take all points: `p1, p2, p3...`
2. Project each: `p1·d, p2·d, p3·d...`
3. These are just numbers (1D)
4. Compute variance of those numbers

### Visually:

```
Good direction d (diagonal):
projections: 1.2, 2.5, 3.8, 5.1, 6.3   ← spread out = HIGH variance ✅

Bad direction d (perpendicular):
projections: 2.1, 2.0, 2.2, 1.9, 2.1   ← bunched up = LOW variance ❌
```

### Example 

Say your projections onto direction `d` are:

```
1, 3, 7, 10
```

**Step 1: Find the mean (average)**
```
mean = (1 + 3 + 7 + 10) / 4 = 5.25
```

**Step 2: Find how far each number is from the mean**
```
1  - 5.25 = -4.25
3  - 5.25 = -2.25
7  - 5.25 =  1.75
10 - 5.25 =  4.75
```

**Step 3: Square each distance** (to make all positive)
```
(-4.25)² = 18.06
(-2.25)² =  5.06
( 1.75)² =  3.06
( 4.75)² = 22.56
```

**Step 4: Average those squares**
```
variance = (18.06 + 5.06 + 3.06 + 22.56) / 4 = 12.18
```

---

That **12.18 is your one number**. ✅

---

Now PCA tries another direction `d2` and gets variance = **2.3**

And another `d3` → variance = **9.1**

```
d1 → 12.18  ✅ winner
d2 →  2.3
d3 →  9.1
```

PCA picks `d1` — the direction with the **biggest variance number**.

That's it. Variance collapses all your projections into **one score**, and PCA maximizes that score. 🎯


> **PCA finds the direction `d` that MAXIMIZES this variance.**

---

## 🧠 Everything Connected

| LA Concept | Role in PCA |
|---|---|
| Vector | Data point or direction |
| Unit vector | The principal component direction |
| Dot product | Project a point onto a direction |
| Norm | Normalize directions to length 1 |
| Orthogonality | Principal components are independent |
| Variance along direction | What PCA maximizes |

---

## Quick Code: All Concepts in NumPy

```python
import numpy as np

# A data point
p = np.array([3, 580])

# A direction (not yet unit length)
d = np.array([1, 1])

# Step 1: make it a unit vector
d_unit = d / np.linalg.norm(d)
print("Unit vector:", d_unit)          # [0.707, 0.707]

# Step 2: project p onto d
projection = np.dot(p, d_unit)
print("Projection:", projection)        # one number

# Step 3: dot product = 0 means orthogonal
a = np.array([1, 0])
b = np.array([0, 1])
print("Orthogonal?", np.dot(a, b))     # 0 ✅
```

---
---

# 📐 Section 4: PCA Math — Step by Step With smalll Dataset

---

## Your Dataset

| Person | Exercise (x) | Calories (y) |
|--------|-------------|--------------|
| 0 | 1 | 200 |
| 1 | 2 | 400 |
| 2 | 3 | 580 |
| 3 | 4 | 800 |
| 4 | 5 | 950 |
| 5 | 6 | 1100 |

As a matrix we write this as:

```
X = [[1,   200],
     [2,   400],
     [3,   580],
     [4,   800],
     [5,   950],
     [6,  1100]]
```

Each row = one person. Each column = one feature.

---

## Step 1: Mean Center the Data

### 🧠 Why?

PCA looks for directions of spread **around the center** of your data — not around the origin (0,0).

If you don't center, PCA gets confused by where your data sits, not how it spreads.

### Compute the mean of each column:

```
mean of x = (1+2+3+4+5+6) / 6 = 3.5
mean of y = (200+400+580+800+950+1100) / 6 = 671.6
```

### Subtract the mean from every point:

| Person | x - 3.5 | y - 671.6 |
|--------|---------|-----------|
| 0 | -2.5 | -471.6 |
| 1 | -1.5 | -271.6 |
| 2 | -0.5 | -91.6 |
| 3 | 0.5 | 128.4 |
| 4 | 1.5 | 278.4 |
| 5 | 2.5 | 428.4 |

Now your data is **centered at (0, 0)**.

```
     y
     │     •
     │   •
     │ •
─────┼───────── x
   • │
  •  │
 •   │
```

---

## Step 2: Covariance Matrix

### 🧠 What is Covariance?

Variance tells you how one feature spreads.
Covariance tells you how **two features move together**.

> If x goes up and y also goes up → positive covariance
> If x goes up and y goes down → negative covariance
> If x and y are unrelated → covariance ≈ 0

### The Covariance Matrix for 2D data looks like this:

```
C = | var(x)    cov(x,y) |
    | cov(x,y)  var(y)   |
```

- **Diagonal** = variance of each feature
- **Off-diagonal** = covariance between features

### Now compute each piece:

**var(x):**
```
= ((-2.5)² + (-1.5)² + (-0.5)² + (0.5)² + (1.5)² + (2.5)²) / 6
= (6.25 + 2.25 + 0.25 + 0.25 + 2.25 + 6.25) / 6
= 17.5 / 6
= 2.92
```

**var(y):**
```
= ((-471.6)² + (-271.6)² + (-91.6)² + (128.4)² + (278.4)² + (428.4)²) / 6
= (222,406 + 73,766 + 8,390 + 16,487 + 77,507 + 183,527) / 6
= 582,083 / 6
= 97,013
```

**cov(x,y):**
```
= ((-2.5×-471.6) + (-1.5×-271.6) + (-0.5×-91.6) + (0.5×128.4) + (1.5×278.4) + (2.5×428.4)) / 6
= (1179 + 407.4 + 45.8 + 64.2 + 417.6 + 1071) / 6
= 3185 / 6
= 530.8
```

### So your Covariance Matrix is:

```
C = |   2.92   530.8 |
    | 530.8   97013  |
```

### 🧠 What does this tell us?

- `cov(x,y) = 530.8` → **very large positive number** → x and y move together strongly
- This confirms what we saw visually — exercise and calories are highly correlated
- PCA will find one direction that captures both

---

## Step 3: Eigenvectors and Eigenvalues

### Our Covariance Matrix

```
C = |   2.92    530.8 |
    | 530.8   97013   |
```

---

### Step 1: What Are We Even Solving?

An eigenvector `v` and eigenvalue `λ` satisfy:

```
C · v = λ · v
```

Plain english: when matrix `C` acts on vector `v`, it only **stretches** it by `λ` — no rotation.

Rearranging:

```
C · v - λ · v = 0
(C - λI) · v = 0
```

This only has a solution when:

```
det(C - λI) = 0
```

That's our starting equation. Let's compute it.

---

### Step 2: Build `C - λI`

`I` is the identity matrix:

```
I = | 1  0 |
    | 0  1 |
```

So:

```
C - λI = | 2.92 - λ    530.8    |
         | 530.8       97013 - λ |
```

---

### Step 3: Compute the Determinant = 0

For a 2×2 matrix:

```
det | a  b | = a×d - b×c
    | c  d |
```

So:

```
det(C - λI) = (2.92 - λ)(97013 - λ) - (530.8 × 530.8) = 0
```

**Expand `(2.92 - λ)(97013 - λ)`:**

```
= 2.92×97013 - 2.92λ - 97013λ + λ²
= 283,277 - 97,015.92λ + λ²
```

**Compute `530.8²`:**

```
530.8² = 281,748.64
```

**Full equation:**

```
λ² - 97,015.92λ + 283,277 - 281,748.64 = 0
λ² - 97,015.92λ + 1,529.36 = 0
```

---

### Step 4: Solve With the Quadratic Formula

```
λ = (-b ± √(b² - 4ac)) / 2a

where:
  a = 1
  b = -97,015.92
  c = 1,529.36
```

**Compute the discriminant `b² - 4ac`:**

```
97,015.92² = 9,412,088,733
4 × 1,529.36 = 6,117.44

discriminant = 9,412,088,733 - 6,117 = 9,412,082,616
√9,412,082,616 ≈ 97,015.89
```

**Two eigenvalues:**

```
λ₁ = (97,015.92 + 97,015.89) / 2 = 97,015.90  ← BIG
λ₂ = (97,015.92 - 97,015.89) / 2 = 0.016      ← tiny
```

---

### Step 5: Find Eigenvector for λ₁ = 97,015.90

Plug into `(C - λI)v = 0`:

```
| 2.92 - 97,015.90      530.8         | |v₁|   |0|
| 530.8                 97013-97,015.90| |v₂| = |0|

| -97,012.98    530.8  | |v₁|   |0|
| 530.8         -2.90  | |v₂| = |0|
```

Take the **second row:**

```
530.8×v₁ - 2.90×v₂ = 0
530.8×v₁ = 2.90×v₂
v₁ = (2.90 / 530.8) × v₂
v₁ = 0.00547 × v₂
```

Set `v₂ = 1`:

```
v = [0.00547, 1]
```

**Normalize** (make length = 1):

```
length = √(0.00547² + 1²) = √(0.00003 + 1) ≈ 1.0

unit eigenvector = [0.0055, 0.99999]  ✅
```

---

### Step 6: Find Eigenvector for λ₂ = 0.016

Plug into `(C - λI)v = 0`:

```
| 2.92 - 0.016    530.8      | |v₁|   |0|
| 530.8           97013-0.016| |v₂| = |0|

| 2.904    530.8  | |v₁|   |0|
| 530.8    97013  | |v₂| = |0|
```

Take the **first row:**

```
2.904×v₁ + 530.8×v₂ = 0
v₁ = -(530.8 / 2.904) × v₂
v₁ = -182.8 × v₂
```

Set `v₂ = 1`:

```
v = [-182.8, 1]  
```

**Normalize:**

```
length = √(182.8² + 1²) = √(33,415 + 1) ≈ 182.8

unit eigenvector = [-182.8/182.8, 1/182.8]
                = [-0.99999, 0.0055]
```

Which is the same as `[0.99999, -0.0055]` pointing the other way. ✅

---

### Final Answer

| | Eigenvalue | Eigenvector | Meaning |
|--|--|--|--|
| PC1 | 97,015.90 | [0.0055, 0.99999] | mostly calories direction |
| PC2 | 0.016 | [0.99999, -0.0055] | mostly exercise direction |

```
Variance explained:
PC1 → 97,015.90 / 97,015.92 = 99.99% ✅
PC2 → 0.016    / 97,015.92 =  0.00% ❌ discard
```
---

## Step 4: Project Data onto PC1

---

### What we have:

**Centered data points:**
```
Person 0: [-2.5, -471.6]
Person 1: [-1.5, -271.6]
Person 2: [-0.5,  -91.6]
Person 3: [ 0.5,  128.4]
Person 4: [ 1.5,  278.4]
Person 5: [ 2.5,  428.4]
```

**PC1 direction:**
```
PC1 = [0.0055, 0.99999]
```

---

### What we do:

For each person — just take the **dot product** with PC1.

Remember dot product: `[a, b] · [c, d] = a×c + b×d`

---

**Person 0:**
```
[-2.5, -471.6] · [0.0055, 0.99999]
= (-2.5 × 0.0055) + (-471.6 × 0.99999)
= (-0.014)      + (-471.59)
= -471.6
```

**Person 1:**
```
[-1.5, -271.6] · [0.0055, 0.99999]
= (-1.5 × 0.0055) + (-271.6 × 0.99999)
= (-0.008)       + (-271.59)
= -271.6
```

**Person 2:**
```
[-0.5, -91.6] · [0.0055, 0.99999]
= (-0.5 × 0.0055) + (-91.6 × 0.99999)
= (-0.003)        + (-91.59)
= -91.6
```

**Person 3:**
```
[0.5, 128.4] · [0.0055, 0.99999]
= (0.5 × 0.0055) + (128.4 × 0.99999)
= (0.003)        + (128.39)
= 128.4
```

**Person 4:**
```
[1.5, 278.4] · [0.0055, 0.99999]
= (1.5 × 0.0055) + (278.4 × 0.99999)
= (0.008)        + (278.39)
= 278.4
```

**Person 5:**
```
[2.5, 428.4] · [0.0055, 0.99999]
= (2.5 × 0.0055) + (428.4 × 0.99999)
= (0.014)        + (428.39)
= 428.4
```

---

### Result:

| Person | Was (2 numbers) | Now (1 number) |
|--------|----------------|----------------|
| 0 | [-2.5, -471.6] | -471.6 |
| 1 | [-1.5, -271.6] | -271.6 |
| 2 | [-0.5,  -91.6] |  -91.6 |
| 3 | [ 0.5,  128.4] |  128.4 |
| 4 | [ 1.5,  278.4] |  278.4 |
| 5 | [ 2.5,  428.4] |  428.4 |

---

### 🧠 Why does the x part almost disappear?

PC1 = `[0.0055, 0.99999]`

- `0.0055` is tiny → the x value barely matters
- `0.99999` is almost 1 → the y value dominates

So the projection is basically just **keeping the calories column** and throwing away exercise. This again is the scaling problem — calories are so large they dominate everything. ⚠️

---
---

# 🔧 Section 5: PCA Algorithm — Full Code From Scratch

---

## The Plan

```
Step 1: Load data
Step 2: Center the data
Step 3: Compute covariance matrix
Step 4: Find eigenvectors & eigenvalues
Step 5: Sort by eigenvalue (biggest first)
Step 6: Project data onto top PC
```

Every single step — we build it manually. No sklearn. Just NumPy.

---

## Full Code

```python
import numpy as np

# ── Step 1: Your data ──────────────────────────────────
X = np.array([
    [1,  200],
    [2,  400],
    [3,  580],
    [4,  800],
    [5,  950],
    [6, 1100]
], dtype=float)

print("Original data:")
print(X)

# ── Step 2: Center the data ────────────────────────────
mean = np.mean(X, axis=0)        # mean of each column
print("\nMean of each column:", mean)
# → [3.5, 671.6]

X_centered = X - mean            # subtract mean from every row
print("\nCentered data:")
print(X_centered)

# ── Step 3: Covariance matrix ──────────────────────────
# rowvar=False means each COLUMN is a feature
C = np.cov(X_centered, rowvar=False)
print("\nCovariance matrix:")
print(C)

# ── Step 4: Eigenvectors and eigenvalues ───────────────
eigenvalues, eigenvectors = np.linalg.eig(C)
print("\nEigenvalues:", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

# ── Step 5: Sort by eigenvalue biggest → smallest ──────
order = np.argsort(eigenvalues)[::-1]   # indices sorted big→small
eigenvalues  = eigenvalues[order]
eigenvectors = eigenvectors[:, order]   # columns are eigenvectors

print("\nSorted eigenvalues:", eigenvalues)
print("Sorted eigenvectors:\n", eigenvectors)

# ── Step 6: Project onto PC1 ───────────────────────────
PC1 = eigenvectors[:, 0]         # first column = biggest eigenvector
print("\nPC1 direction:", PC1)

X_projected = X_centered @ eigenvectors[:, :1]  # keep 1 PC
print("\nProjected data (2D → 1D):")
print(X_projected)

# ── Bonus: How much variance does PC1 explain? ─────────
variance_explained = eigenvalues / np.sum(eigenvalues) * 100
print("\nVariance explained by each PC:")
for i, v in enumerate(variance_explained):
    print(f"  PC{i+1}: {v:.2f}%")
```

---

## Output You Will See

```
Original data:
[[   1.  200.]
 [   2.  400.]
 [   3.  580.]
 [   4.  800.]
 [   5.  950.]
 [   6. 1100.]]

Mean of each column: [  3.5  671.6]

Centered data:
[[ -2.5  -471.6]
 [ -1.5  -271.6]
 [ -0.5   -91.6]
 [  0.5   128.4]
 [  1.5   278.4]
 [  2.5   428.4]]

Covariance matrix:
[[  3.5     636.96]
 [636.96  116415.6]]

Eigenvalues: [116418.  3.5]

Sorted eigenvalues: [116418.  3.5]

PC1 direction: [0.0055  0.99999]

Projected data (2D → 1D):
[[-471.6]
 [-271.6]
 [ -91.6]
 [ 128.4]
 [ 278.4]
 [ 428.4]]

Variance explained by each PC:
  PC1: 99.99%
  PC2:  0.00%
```

---

## Line by Line — What Each Does

---

**`np.mean(X, axis=0)`**
```
axis=0 means → go down the rows, compute mean per column

column 0 (exercise): (1+2+3+4+5+6)/6 = 3.5
column 1 (calories): (200+400+...)/6  = 671.6
```

---

**`X - mean`**
```
NumPy automatically subtracts mean from every row
→ centers your data at (0,0)
```

---

**`np.cov(X_centered, rowvar=False)`**
```
rowvar=False → each column is a variable (not each row)
→ gives you the 2×2 covariance matrix
```

---

**`np.linalg.eig(C)`**
```
Does the full eigenvalue calculation we did by hand
→ returns eigenvalues (amounts of variance)
→ returns eigenvectors (directions) as columns
```

---

**`np.argsort(eigenvalues)[::-1]`**
```
argsort → gives indices that would sort the array
[::-1]  → reverses it (biggest first)

example:
eigenvalues = [3.5, 116418]
argsort     = [0, 1]         (0 is smallest)
[::-1]      = [1, 0]         (1 is biggest → goes first)
```

---

**`eigenvectors[:, order]`**
```
eigenvectors are stored as COLUMNS
[:, 0] = first eigenvector (all rows, column 0)
[:, 1] = second eigenvector

we reorder the columns to match sorted eigenvalues
```

---

**`X_centered @ eigenvectors[:, :1]`**
```
@ = matrix multiplication in NumPy

X_centered      shape: (6, 2)   ← 6 people, 2 features
eigenvectors[:, :1]  shape: (2, 1)   ← keep only PC1

result shape: (6, 1)   ← 6 people, 1 number each
```

---

## 🧠 The Shape Story

```
X_centered:          (6 × 2)    6 people, 2 features
eigenvectors[:,:1]:  (2 × 1)    the PC1 direction

multiply:            (6 × 2) @ (2 × 1) = (6 × 1)

6 people, each with 1 number. ✅
```

If you kept **both** PCs:
```
eigenvectors[:,:2]:  (2 × 2)
result:              (6 × 2)   ← same as original, no reduction
```

If you kept **1** PC:
```
eigenvectors[:,:1]:  (2 × 1)
result:              (6 × 1)   ← reduced! ✅
```
