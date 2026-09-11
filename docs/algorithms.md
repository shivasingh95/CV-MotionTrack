# Computer Vision Algorithms: CV-MotionTrack

This document details the theoretical concepts and mathematical foundations of the computer vision algorithms utilized in **CV-MotionTrack** for the CSE3010 Computer Vision curriculum.

---

## 1. Image Preprocessing Fundamentals

Image preprocessing constitutes the primary foundational stage of any video analysis pipeline. High-speed video sensors capture frames subject to electronic thermal noise, photon shot noise, compression artifacts, and ambient illumination fluctuations. Without disciplined preprocessing, these high-frequency perturbations propagate into differential operators and statistical models, causing false-positive detections and fragmented motion trajectories.

### 1.1 Grayscale Intensity Conversion

#### Concept and Motivation
Video streams are typically captured in 3-channel color representations, such as RGB or BGR, where each pixel is defined by a triplet $(R, G, B)$ representing spectral intensity in the red, green, and blue wavelengths. While color information is beneficial for chromatic segmentation and semantic scene classification, motion detection and spatial gradient estimation fundamentally depend on changes in **luminance** (radiant energy weighted by human visual sensitivity) across space and time.

Grayscale conversion collapses the 3D color cube into a 1D scalar intensity field:
$$I: \mathbb{Z}^2 \to [0, 255]$$

#### Why Grayscale Conversion is Used:
1. **Computational Efficiency**: Reduces memory footprint and per-pixel processing operations by a factor of 3 (from 3 bytes to 1 byte per pixel), which is critical for real-time frame rates.
2. **Elimination of Chromatic Ambiguity**: Color differences often reflect surface pigment rather than geometry or physical motion. Luminance changes capture true object displacement and edge contours.
3. **Compatibility with Classical Operators**: Standard differential operators (spatial derivatives $\nabla I$, Hessian matrices, and Laplacian filters) and statistical background models (e.g., Gaussian mixture models) are formulated on scalar intensity fields.

#### Mathematical Formulation (ITU-R BT.601)
In OpenCV, conversion from BGR to Grayscale uses the standard psychophysical luminance weighting:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$
where green receives the highest weighting due to human retinal sensitivity peaking near 555 nm.

---

### 1.2 Gaussian Spatial Filtering

#### Concept and Motivation
Raw captured images contain high-frequency noise characterized by sudden, uncorrelated pixel intensity spikes. Because moving object detection relies on computing spatial derivatives and temporal frame differences, noise creates localized high-gradient artifacts:
$$\frac{\partial I}{\partial x}, \quad \frac{\partial I}{\partial y}, \quad \frac{\partial I}{\partial t}$$

Gaussian filtering performs a spatially weighted local averaging that smooths intensity variations while maintaining rotational symmetry and avoiding directional ringing artifacts.

#### The Continuous 2D Gaussian Kernel
The isotropic two-dimensional Gaussian distribution centered at the origin is defined as:
$$G(x, y) = \frac{1}{2\pi\sigma^2} \exp\left(-\frac{x^2 + y^2}{2\sigma^2}\right)$$

where:
- $(x, y)$ are the spatial coordinates relative to the center of the kernel.
- $\sigma$ is the standard deviation (spread parameter) of the Gaussian distribution.
- $\frac{1}{2\pi\sigma^2}$ is the normalization factor ensuring $\iint_{\mathbb{R}^2} G(x, y) \, dx \, dy = 1$.

#### Properties of the Gaussian Filter
1. **Weighted Smoothing**: Unlike a simple box filter (which assigns uniform weights $\frac{1}{k^2}$ to all neighborhood pixels), the Gaussian filter assigns maximum weight to the central pixel, with weights decaying smoothly as a function of Euclidean distance $r = \sqrt{x^2 + y^2}$. This preserves spatial locality and minimizes artificial edge artifacts.
2. **Rotational Symmetry (Isotropy)**: $G(x, y)$ is radially symmetric, meaning the smoothing behavior is invariant to directional orientation.
3. **Separability**: The 2D Gaussian kernel factors into two independent 1D convolutions:
   $$G(x, y) = G_{1D}(x) \cdot G_{1D}(y) = \left(\frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{x^2}{2\sigma^2}\right)\right) \left(\frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{y^2}{2\sigma^2}\right)\right)$$
   This reduces the computational complexity of a $K \times K$ kernel convolution from $\mathcal{O}(K^2)$ multiplications per pixel to $\mathcal{O}(2K)$.
4. **Frequency Domain Attenuation**: The Fourier transform of a Gaussian is also a Gaussian. In the frequency domain, it functions as an ideal low-pass filter with no sidelobes, effectively eliminating high-frequency sensor noise without creating oscillatory Gibbs phenomena.

#### Discrete Approximation and Kernel Sizing
In digital image processing, $G(x, y)$ is sampled onto a discrete matrix of dimensions $(2k+1) \times (2k+1)$ (positive odd integers, such as $3 \times 3$, $5 \times 5$, or $7 \times 7$):
$$I_{\text{smooth}}(x, y) = \sum_{i=-k}^{k} \sum_{j=-k}^{k} G_{\text{discrete}}(i, j) \cdot I(x - i, y - j)$$
When $\sigma = 0$ is specified in OpenCV, $\sigma$ is automatically derived from kernel dimensions:
$$\sigma = 0.3 \cdot \left(\frac{K - 1}{2} - 1\right) + 0.8$$

---

### 1.3 Why Preprocessing is Essential Before Object Detection

In a classical motion detection system, background subtraction identifies moving pixels by computing the discrepancy between the incoming frame $I_t(x, y)$ and a statistical background model $B_t(x, y)$:
$$\Delta I_t(x, y) = |I_t(x, y) - B_t(x, y)|$$

Without grayscale conversion and Gaussian filtering:
1. **False-Positive Noise Pixels**: Sensor noise generates random intensity fluctuations that surpass the background subtraction threshold $\tau$, manifesting as hundreds of isolated "salt-and-pepper" foreground pixels.
2. **Boundary Irregularity**: High-frequency edge noise distorts object contours, causing contour extraction to yield fragmented, erratic bounding boxes.
3. **Downstream Tracker Confusion**: Centroid tracking depends on stable geometric moments. Noise-induced contour jitter directly destabilizes object centroids, causing false velocity spikes in motion analysis.

Applying Gaussian smoothing prior to background modeling ensures that only coherent, spatially contiguous pixel clusters corresponding to genuine moving physical objects are segmented.

---

## 2. Downstream Algorithms (Planned for Subsequent Steps)

The following classical algorithms interface directly with the preprocessed frames:

### 2.1 Background Modeling & Subtraction (Step 4)
- **Gaussian Mixture Models (MOG2)**: Models each pixel's history as a mixture of $K$ adaptive Gaussians to accommodate multi-modal backgrounds (e.g., swaying leaves, monitor flicker).
- **Morphological Filtering**: Structuring element operations:
  - **Erosion**: $\mathcal{A} \ominus \mathcal{B} = \{z \mid (\mathcal{B})_z \subseteq \mathcal{A}\}$
  - **Dilation**: $\mathcal{A} \oplus \mathcal{B} = \{z \mid (\hat{\mathcal{B}})_z \cap \mathcal{A} \neq \emptyset\}$
  - **Opening**: $(\mathcal{A} \ominus \mathcal{B}) \oplus \mathcal{B}$ (eliminates background spackle).
  - **Closing**: $(\mathcal{A} \oplus \mathcal{B}) \ominus \mathcal{B}$ (bridges intra-object voids).

### 2.2 Object Tracking (Step 5)
- **Centroid Association**: Greedy Euclidean bipartite matching between existing track states and new frame detection centroids.
- **State Machine**: Management of track births, sustained trajectory trails, and deregistrations after prolonged disappearance.

### 2.3 Optical Flow & KLT Tracking (Step 6)
- **Shi-Tomasi Corner Detector**: Evaluates the minimum eigenvalue of the spatial structure tensor:
  $$M = \sum \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}, \quad R = \min(\lambda_1, \lambda_2) > \lambda_{\text{threshold}}$$
- **Lucas-Kanade Differential Flow**: Solves the optical flow brightness constancy equation under local spatial coherence assumptions.

### 2.4 Motion Kinematics (Step 7)
- Discrete spatial displacement $\Delta d$, angular heading $\theta = \text{atan2}(\Delta y, \Delta x)$, and velocity $v = \frac{\Delta d}{\Delta t}$ estimation.

---

## 3. Background Modeling and Subtraction (Step 4 – Implemented)

### 3.1 What Background Modeling Means

Background modeling is the process of constructing a statistical representation of the **static (non-moving) scene** from a sequence of video frames. The model captures the expected pixel intensity distribution for each spatial location $(x, y)$ in the absence of moving objects. Once the background model is established, any incoming frame can be compared against it; pixels whose intensities deviate significantly from the model are classified as **foreground** — i.e., they belong to something that has moved or appeared in the scene.

### 3.2 Why Background Subtraction Is Useful for Moving-Object Detection

Classical foreground segmentation by background subtraction is:

- **Computationally cheap**: The decision per pixel is a single arithmetic comparison against a learned model rather than an expensive forward pass through a neural network.
- **Training-data-free**: No labelled dataset is needed; the model is built in real time from the first few seconds of video.
- **Interpretable**: The intermediate foreground mask is a binary image that can be directly inspected and qualitatively validated.
- **Well-suited to fixed or near-fixed cameras**: Surveillance cameras, traffic monitoring systems, and lab setups all benefit from a stable background assumption.

### 3.3 Gaussian Mixture Model Background Subtraction (MOG2)

The OpenCV implementation used in this project, `cv2.createBackgroundSubtractorMOG2`, is based on the work of Zivkovic (2004). It models the temporal intensity history of every pixel $(x, y)$ as a **Mixture of $K$ Gaussians**:

$$P(I_t(x,y)) = \sum_{k=1}^{K} \omega_k \cdot \mathcal{N}\!\left(I_t(x,y);\, \mu_k,\, \sigma_k^2\right)$$

where:
- $K$ is the number of Gaussian components (typically 3–7, chosen adaptively).
- $\omega_k$ is the mixture weight (relative frequency) of component $k$.
- $\mu_k, \sigma_k^2$ are the mean and variance of component $k$, updated online with each new frame.

A pixel is classified as **background** if its intensity is well-explained by one of the $K$ Gaussians (i.e., lies within $\sqrt{\text{varThreshold}}$ standard deviations of any component mean). Otherwise it is classified as **foreground**.

The **`history`** parameter controls the effective time window: a shorter history makes the model adapt faster to environmental changes (e.g., moving curtains) but is more susceptible to sudden noise.

### 3.4 What a Foreground Mask Represents

The output of `apply_background_subtraction(frame)` is a single-channel `uint8` image of the same spatial dimensions as the input:

| Pixel value | Meaning |
|---|---|
| `0`   | Background — pixel matches the learned background model. |
| `127` | Shadow — partial illumination change detected (if `detectShadows=True`). |
| `255` | Foreground — pixel does not match background; belongs to a moving object. |

This three-valued raw mask is the direct output of the Gaussian mixture evaluation.

### 3.5 Why Thresholding Is Needed

MOG2 shadow detection is valuable for robustness, but shadow pixels (value `127`) are *not* moving objects — they are simply regions where an object casts a shadow on the background. Passing shadow pixels into contour analysis would produce spurious bounding boxes around shadow shapes.

A binary threshold at `pixel_value > 127` converts the ternary mask to a strict binary mask `{0, 255}`:

$$M_{\text{binary}}(x,y) = \begin{cases} 255 & \text{if } M_{\text{raw}}(x,y) > 127 \\ 0 & \text{otherwise} \end{cases}$$

This discards all shadow pixels and retains only genuine foreground detections.

### 3.6 Why Morphological Opening and Closing Are Used

Even after thresholding, the binary mask contains two categories of artifact that degrade downstream contour analysis:

**Problem 1 — Salt-and-pepper noise**  
Random foreground pixels caused by camera sensor noise, JPEG compression, or rapid illumination flicker.  
**Solution — Morphological Opening** (erosion followed by dilation):

$$M_{\text{open}} = (M_{\text{binary}} \ominus B) \oplus B$$

Erosion $\ominus$ shrinks every foreground region.  Isolated noise pixels, which are narrower than the structuring element $B$, disappear entirely.  The subsequent dilation $\oplus$ restores the boundary of the remaining (genuine) regions to approximately their original position.

**Problem 2 — Intra-object holes and gaps**  
Moving objects may have regions of low texture (e.g., a uniformly coloured t-shirt) that do not differ from the background, creating dark holes inside the foreground blob.  These cause a single object to produce multiple small contours.  
**Solution — Morphological Closing** (dilation followed by erosion):

$$M_{\text{close}} = (M_{\text{open}} \oplus B) \ominus B$$

Dilation $\oplus$ expands the foreground regions, bridging nearby gaps.  The subsequent erosion $\ominus$ shrinks them back, leaving the outer boundary roughly intact while the interior holes are filled.

Both operations use a small rectangular structuring element (default $3 \times 3$) whose size is configurable via `MORPH_KERNEL_SIZE` in `config.py`.

### 3.7 How Contours Are Used to Obtain Object Regions

After morphological cleaning the mask is a smooth binary image where connected white blobs represent candidate moving objects.  `cv2.findContours` with `cv2.RETR_EXTERNAL` traces the outer boundary of each blob using border-following.

For each contour:

1. **Area filter** — contours with pixel area $< \text{MIN\_OBJECT\_AREA}$ or $> \text{MAX\_OBJECT\_AREA}$ are discarded.
2. **Bounding rectangle** — `cv2.boundingRect(contour)` returns $(x, y, w, h)$.
3. **Centroid from image moments** — the centroid $(c_x, c_y)$ is more accurate than the bounding-rect centre for non-rectangular blobs:
   $$c_x = \frac{M_{10}}{M_{00}}, \qquad c_y = \frac{M_{01}}{M_{00}}$$
   where $M_{pq} = \sum_{x} \sum_{y} x^p y^q \cdot I(x,y)$ are spatial image moments.
4. A `Detection` dataclass instance is created and appended to the result list.

### 3.8 Conceptual Workflow Summary

```
Raw video frame  (H × W × 3, BGR)
        │
        ▼
Preprocessor — grayscale + Gaussian smoothing
        │
        ▼  preprocessed frame (H × W, uint8)
        │
        ▼
Background Model (MOG2 GMM, online update)
        │
        ▼  raw foreground mask {0, 127, 255}
        │
        ▼
Threshold at 127 → binary mask {0, 255}
        │
        ▼
Morphological Opening  (remove speckle noise)
        │
        ▼
Morphological Closing  (fill intra-object gaps)
        │
        ▼  clean binary mask
        │
        ▼
cv2.findContours → filter by area
        │
        ▼
List[Detection]  (bbox, centroid, area)
        │
        ▼ (next step)
ObjectTracker
```

---

## 4. Optical Flow and Motion Estimation (Step 6 – Implemented)

### 4.1 Optical Flow Fundamentals

#### What Optical Flow Means
**Optical flow** is the pattern of apparent motion of image intensity patterns between two consecutive frames in a visual sequence. It represents the velocity field $(\mathbf{u}, \mathbf{v}) = \left(\frac{dx}{dt}, \frac{dy}{dt}\right)$ that describes how each visual pattern moves across the 2D image plane over time:
$$\mathbf{v}(x, y) = [u(x, y), v(x, y)]^T$$

#### What Apparent Motion Means
Optical flow describes **apparent motion** — the observable movement of brightness patterns across the image plane — which is not always identical to the physical 3D motion of objects in the world:
1. **Apparent motion without physical motion**: A stationary textured sphere illuminated by a moving light source produces changing shadow and specular highlights, resulting in non-zero optical flow despite the object being physically at rest.
2. **Physical motion without apparent motion**: A perfectly smooth, uniformly colored sphere rotating about its axis produces zero optical flow because its surface intensity distribution does not change across consecutive frames.

In real-world computer vision systems (such as surveillance and vehicle tracking), surface texture is non-uniform, meaning apparent pixel motion correlates strongly with true scene kinematics.

#### Why Consecutive Frames Are Used
Optical flow relies on differential image analysis between consecutive video frames captured at times $t$ and $t + \Delta t$. When $\Delta t$ is small (typical video frame rates of 25–60 FPS):
- Inter-frame spatial displacements are small ($\approx 1\text{--}5$ pixels).
- First-order Taylor series approximations remain valid.
- Scene illumination and geometric appearance remain locally invariant.

#### Sparse vs. Dense Optical Flow
| Property | Sparse Optical Flow (Lucas-Kanade) | Dense Optical Flow (Farnebäck / Horn-Schunck) |
|---|---|---|
| **Coverage** | Evaluates motion only at selected high-gradient feature points (e.g., corners). | Computes a motion vector for every single pixel in the entire frame. |
| **Computational Cost** | Extremely low ($\mathcal{O}(N)$ where $N \sim 100$ points); ideal for real-time systems. | Computationally intensive ($\mathcal{O}(W \times H)$ where $W \times H \sim 10^5\text{--}10^6$ pixels). |
| **Reliability** | Highly reliable because tracking is restricted to well-conditioned image points. | Prone to noise and ambiguity in flat or low-texture regions. |
| **Project Fit** | Used in CV-MotionTrack for efficient feature-level tracking and trajectory support. | Excessive for real-time tracking on standard CPU hardware. |

---

### 4.2 The Lucas-Kanade Method

The Lucas-Kanade algorithm (1981) is a classical differential method for sparse optical flow estimation.

#### 1. Brightness Constancy Assumption
The fundamental premise of optical flow is that the intensity of a moving scene point remains constant across consecutive time instants:
$$I(x, y, t) \approx I(x + u, y + v, t + 1)$$

#### 2. Linearization via First-Order Taylor Series Expansion
Assuming small displacements $(u, v)$ over time interval $\Delta t = 1$, we expand the right-hand side using a 2D first-order Taylor series:
$$I(x + u, y + v, t + 1) \approx I(x, y, t) + \frac{\partial I}{\partial x} u + \frac{\partial I}{\partial y} v + \frac{\partial I}{\partial t}$$

Subtracting $I(x, y, t)$ yields the **Optical Flow Constraint Equation**:
$$I_x u + I_y v + I_t = 0$$
or in vector notation:
$$\nabla I \cdot [u, v]^T = -I_t$$
where:
- $I_x = \frac{\partial I}{\partial x}$ is the horizontal spatial image gradient.
- $I_y = \frac{\partial I}{\partial y}$ is the vertical spatial image gradient.
- $I_t = \frac{\partial I}{\partial t}$ is the temporal image gradient between consecutive frames.
- $u = \frac{dx}{dt}, v = \frac{dy}{dt}$ are the unknown horizontal and vertical velocity components.

#### 3. Why One Equation Is Insufficient (The Aperture Problem)
The optical flow constraint is a single linear scalar equation with two unknown variables $(u, v)$. Consequently, an infinite number of velocity vectors satisfy the equation:
$$I_x u + I_y v = -I_t$$
We can only determine motion **perpendicular** to the edge orientation (along the gradient direction $\nabla I$). The component of motion parallel to the edge is unobservable through a small local window — a fundamental limitation known as the **aperture problem**.

#### 4. The Local Spatial Coherence Assumption
Lucas and Kanade resolved the aperture problem by introducing a local spatial coherence constraint:
> *All pixels within a small spatial window $\Omega$ of size $W \times W$ (e.g., $21 \times 21$) centered at point $p$ share identical velocity $(u, v)$.*

For a window of $n = W \times W$ pixels $\{p_1, p_2, \dots, p_n\}$, each pixel provides an instance of the optical flow constraint equation:
$$\begin{cases}
I_x(p_1) u + I_y(p_1) v = -I_t(p_1) \\
I_x(p_2) u + I_y(p_2) v = -I_t(p_2) \\
\vdots \\
I_x(p_n) u + I_y(p_n) v = -I_t(p_n)
\end{cases}$$

Writing this in matrix form $A \mathbf{v} = \mathbf{b}$:
$$A = \begin{bmatrix}
I_x(p_1) & I_y(p_1) \\
I_x(p_2) & I_y(p_2) \\
\vdots & \vdots \\
I_x(p_n) & I_y(p_n)
\end{bmatrix}, \quad
\mathbf{v} = \begin{bmatrix} u \\ v \end{bmatrix}, \quad
\mathbf{b} = -\begin{bmatrix}
I_t(p_1) \\
I_t(p_2) \\
\vdots \\
I_t(p_n)
\end{bmatrix}$$

#### 5. Motion Estimation via Normal Equations
Since $n > 2$, the system is overdetermined. The optimal least-squares solution minimizes the residual $\|A \mathbf{v} - \mathbf{b}\|^2$:
$$(A^T A) \mathbf{v} = A^T \mathbf{b}$$
$$\mathbf{v} = (A^T A)^{-1} A^T \mathbf{b}$$

Expanding the $2 \times 2$ structure tensor $M = A^T A$:
$$A^T A = \begin{bmatrix}
\sum_{i=1}^n I_x^2(p_i) & \sum_{i=1}^n I_x(p_i) I_y(p_i) \\
\sum_{i=1}^n I_x(p_i) I_y(p_i) & \sum_{i=1}^n I_y^2(p_i)
\end{bmatrix}$$

The system has a stable, invertible solution if and only if $A^T A$ has full rank ($\text{rank} = 2$), meaning both eigenvalues $\lambda_1, \lambda_2$ are sufficiently large and well-conditioned. This condition is satisfied precisely at **corner feature points**.

---

### 4.3 Pyramidal Lucas-Kanade

#### Large Displacement Problem
The first-order Taylor expansion assumes pixel displacements are infinitesimal ($|u|, |v| \le 1\text{--}2\text{ pixels}$). When objects move quickly across the video frame, inter-frame displacement may be $10\text{--}30\text{ pixels}$, violating the linearization assumption and causing standard Lucas-Kanade to fail to converge.

#### Multi-Scale Gaussian Pyramids (`max_level`)
Pyramidal Lucas-Kanade (Bouguet, 2001) resolves this limitation through multi-scale coarse-to-fine processing:
1. **Pyramid Construction**: A Gaussian image pyramid of depth $L = \text{max\_level}$ (default $L=3$) is constructed for each frame by iteratively downsampling by a factor of 2:
   $$\text{Level } 0: W \times H \quad (\text{full resolution})$$
   $$\text{Level } 1: \frac{W}{2} \times \frac{H}{2}$$
   $$\text{Level } 2: \frac{W}{4} \times \frac{H}{4}$$
   $$\text{Level } 3: \frac{W}{8} \times \frac{H}{8} \quad (\text{coarsest level})$$
2. **Coarse Scale Estimation**: A large displacement of 24 pixels at Level 0 becomes a displacement of only $24 / 2^3 = 3$ pixels at Level 3, which is well within the convergence range of Lucas-Kanade.
3. **Coarse-to-Fine Propagation**: The estimated flow vector from Level $L$ is scaled by 2 and passed as an initial motion estimate to Level $L-1$. At each level, Lucas-Kanade computes a small residual displacement vector, refining the estimate down to Level 0.

This enables tracking of fast-moving objects while preserving sub-pixel precision at the original image resolution.

---

### 4.4 KLT-Style Feature Tracking

#### Corner Detection (Shi-Tomasi Algorithm)
Kanade-Lucas-Tomasi (KLT) tracking couples the Lucas-Kanade motion solver with the Shi-Tomasi corner detector (`cv2.goodFeaturesToTrack`):
- Flat regions: $\lambda_1 \approx 0, \lambda_2 \approx 0$ (untrackable, singular matrix).
- Edges: $\lambda_1 \gg 0, \lambda_2 \approx 0$ (aperture problem, ill-conditioned).
- Corners: $\lambda_1 \ge \lambda_{\min} > 0, \lambda_2 \ge \lambda_{\min} > 0$ (well-conditioned, trackable in both $x$ and $y$).

Shi and Tomasi (1994) defined the corner response metric as:
$$R = \min(\lambda_1, \lambda_2)$$
Points with $R > \text{quality\_level} \cdot \max(R)$ and minimum mutual distance $\text{min\_distance}$ are selected as initial tracking candidates.

#### Tracking Lifecycle & Automatic Re-Detection
Feature points naturally degrade over time:
- Objects rotate or undergo non-rigid deformations.
- Tracked features exit the camera field of view.
- Occlusions cause tracking convergence failure (`status == 0`).

`OpticalFlowAnalyzer` monitors the number of active, valid points. When the active point count drops below `OPTICAL_FLOW_MIN_FEATURES` (default 10), new Shi-Tomasi corners are automatically detected and added to the tracking pool, guaranteeing uninterrupted motion tracking across extended video sequences.

*(Note: CV-MotionTrack utilizes OpenCV's production-grade C++ implementations `cv2.goodFeaturesToTrack` and `cv2.calcOpticalFlowPyrLK` wrapped within a clean, modular Python architecture, rather than an unoptimized from-scratch reimplementation).*

---

## 5. Motion Analysis (Step 7 – Implemented)

### 5.1 Why Object Trajectories Are Analyzed

In automated visual surveillance, traffic management, and athletic kinematic tracking, detecting an object in a single isolated frame is insufficient. A single detection reveals **where** an entity is, but yields zero insight into **what** it is doing.

Analyzing object trajectories:
1. **Discloses Behavioral Dynamics**: Distinguishes purposeful directional motion (e.g., pedestrian crossing a street, vehicle executing a turn) from stationary background clutter or localized jitter (e.g., leaves rustling).
2. **Smooths Measurement Inaccuracies**: Momentary boundary variations in contour segmentation cause centroid fluctuations. Trajectory analysis allows rolling moving-average filters to dampen high-frequency jitter.
3. **Enables Predictive Kinematics**: Historical position sequences allow estimating instantaneous velocities, predicting future positions, and maintaining identity through brief visual occlusions.

---

### 5.2 Position vs. Displacement vs. Path Length

A critical conceptual distinction in classical Computer Vision kinematics:

#### 1. Position
An absolute 2D coordinate in the digital image coordinate frame at a specific temporal epoch $t$:
$$\mathbf{p}(t) = (x(t), y(t)) \in \mathbb{R}^2$$
Position indicates spatial location relative to the image sensor's top-left origin $(0, 0)$.

#### 2. Frame-to-Frame Displacement
The differential vector and Euclidean magnitude between consecutive temporal positions $\mathbf{p}(t_1) = (x_1, y_1)$ and $\mathbf{p}(t_2) = (x_2, y_2)$:
$$dx = x_2 - x_1$$
$$dy = y_2 - y_1$$
$$d = \sqrt{dx^2 + dy^2}$$
Displacement measures the immediate, local spatial step taken over time interval $\Delta t = t_2 - t_1$.

#### 3. Path Length vs. Net Displacement
For an object trajectory consisting of $n$ historical centroid observations:
$$T = [\mathbf{p}_1, \mathbf{p}_2, \dots, \mathbf{p}_n] = [(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)]$$

- **Total Path Length ($L$)**: The cumulative distance traveled along the entire historical path:
  $$L = \sum_{i=1}^{n-1} \sqrt{(x_{i+1} - x_i)^2 + (y_{i+1} - y_i)^2}$$
  $L$ reflects the total mechanical distance covered, including all curves, loops, and wanderings.

- **Net Displacement ($D$)**: The direct straight-line Euclidean distance from the starting coordinate to the ending coordinate:
  $$D = \|\mathbf{p}_n - \mathbf{p}_1\| = \sqrt{(x_n - x_1)^2 + (y_n - y_1)^2}$$
  $D$ reflects overall spatial relocation.

```
       Path Length (L = 10 + 10 = 20.0 px)
       (0, 0) ───────────────────► (10, 0)
                                      │
                                      │
                                      ▼
                                   (10, 10)
       Net Displacement (D = sqrt(10² + 10²) ≈ 14.14 px)
       (0, 0) - - - - - - - - - -► (10, 10)
```

By the triangle inequality, $L \ge D$. The ratio $\eta = \frac{D}{L} \in [0, 1]$ serves as a measure of path tortuosity or directional efficiency:
- $\eta = 1$: Perfectly straight, rectilinear trajectory.
- $\eta \approx 0$: Wandering, looping, or stationary hovering motion.

---

### 5.3 Direction and Heading Angle Estimation

#### Coordinate System Convention
Digital image sensors employ an inverted vertical coordinate convention:
- **X-axis**: Points horizontally to the **RIGHT** ($+X$).
- **Y-axis**: Points vertically **DOWNWARD** ($+Y$).

Consequently:
- An object moving upward exhibits **negative** $dy$ ($y_2 < y_1$).
- An object moving downward exhibits **positive** $dy$ ($y_2 > y_1$).
- An object moving rightward exhibits **positive** $dx$ ($x_2 > x_1$).
- An object moving leftward exhibits **negative** $dx$ ($x_2 < x_1$).

#### Stationary Threshold Suppression
If the frame-to-frame displacement magnitude is smaller than the configurable stationary threshold:
$$d = \sqrt{dx^2 + dy^2} < \text{STATIONARY\_THRESHOLD} \quad (\text{default } 1.0\text{ px})$$
the motion is classified as **`STATIONARY`**. This prevents minor sub-pixel contour variations from generating false directional flags.

#### Qualitative Directional Sector Classification
For displacements exceeding the threshold, heading angle is computed using the four-quadrant inverse tangent:
$$\theta = \text{atan2}(dy, dx) \pmod{360^\circ}$$
The $360^\circ$ circle is partitioned into 8 continuous $45^\circ$ directional cones:

| Sector Angle Range | Qualitative Direction | Vector Characteristics |
|---|---|---|
| $[337.5^\circ, 360^\circ) \cup [0^\circ, 22.5^\circ)$ | **RIGHT** | $+dx$, $dy \approx 0$ |
| $[22.5^\circ, 67.5^\circ)$ | **DOWN-RIGHT** | $+dx$, $+dy$ |
| $[67.5^\circ, 112.5^\circ)$ | **DOWN** | $dx \approx 0$, $+dy$ |
| $[112.5^\circ, 157.5^\circ)$ | **DOWN-LEFT** | $-dx$, $+dy$ |
| $[157.5^\circ, 202.5^\circ)$ | **LEFT** | $-dx$, $dy \approx 0$ |
| $[202.5^\circ, 247.5^\circ)$ | **UP-LEFT** | $-dx$, $-dy$ |
| $[247.5^\circ, 292.5^\circ)$ | **UP** | $dx \approx 0$, $-dy$ |
| $[292.5^\circ, 337.5^\circ)$ | **UP-RIGHT** | $+dx$, $-dy$ |

---

### 5.4 Image-Space Velocity

#### Mathematical Definition
Image-space velocity defines the rate of change of 2D spatial position across time:
$$v = \frac{d}{\Delta t} = \frac{\sqrt{dx^2 + dy^2}}{\Delta t}$$

- **Default Interval ($\Delta t = 1\text{ frame}$)**:
  $$v \text{ is expressed in } \textbf{pixels/frame}$$
  This unit is hardware-independent and directly reflects inter-frame pixel displacement.
- **Time-Scaled Interval ($\Delta t = 1 / \text{FPS}$)**:
  $$v_{\text{sec}} = v \cdot \text{FPS} \text{ in } \textbf{pixels/second}$$

#### Critical Academic Limitation: Absence of Real-World Scale
> **Important Note for CSE3010**:  
> Monocular uncalibrated cameras suffer from projective depth ambiguity: an object moving at $10\text{ m/s}$ at a distance of $50\text{ m}$ may produce identical image-space pixel displacement to an object moving at $1\text{ m/s}$ at a distance of $5\text{ m}$.  
> Without camera intrinsic/extrinsic calibration and physical scene scale constraints (e.g., ground-plane homography), velocities **CANNOT** be reported in physical units ($\text{m/s}$ or $\text{km/h}$). CV-MotionTrack strictly reports velocity in **pixels/frame** and **pixels/second**.

---

### 5.5 Motion History & Moving-Average Smoothing

#### Bounded History Management
To avoid memory growth during extended video streams, `MotionAnalyzer` maintains a FIFO rolling queue per active object ID, bounded by `MOTION_HISTORY_LENGTH` (default 30 observations).

#### Rolling Moving-Average Smoothing
Centroid estimates from contour analysis can introduce high-frequency measurement noise. Instantaneous velocities are smoothed using a rolling window of size $K = \text{MOTION\_SMOOTHING\_WINDOW}$ (default 5 frames):
$$\bar{v}_t = \frac{1}{\min(K, N)} \sum_{j=0}^{\min(K, N)-1} v_{t-j}$$
This dampens transient contour vibrations while preserving authentic macro-scale motion trajectories.

---

## 6. Spatio-Temporal Analysis

### Combining Space and Time in Video Analysis

A key conceptual pillar of classical Computer Vision is understanding that video processing is fundamentally **spatio-temporal**:

1. **Spatial Dimension ($\mathbb{R}^2$)**:
   Each frame $I(x, y)$ provides a 2D spatial intensity map capturing object shape, edges, textures, and contour geometry at a frozen instant in time.
2. **Temporal Dimension ($\mathbb{R}^1$)**:
   The chronological sequencing of frames $t_1, t_2, \dots, t_n$ sampled at discrete temporal intervals $\Delta t$ introduces the dimension of time.

```
       Spatial Plane (x, y)
             ▲
             │       Frame t+2: (x3, y3)
             │          ▲
             │          │  Trajectory vector
             │       Frame t+1: (x2, y2)
             │          ▲
             │          │
             │       Frame t: (x1, y1)
             └──────────────────────► Temporal Dimension (t)
```

### Spatio-Temporal Trajectory Representation
A tracked object's trajectory is represented as a spatio-temporal path:
$$\mathcal{T} = \{ (\mathbf{p}_k, t_k) \}_{k=1}^N = \{ (x_1, y_1, t_1), (x_2, y_2, t_2), \dots, (x_N, y_N, t_N) \}$$

By operating simultaneously across both space and time:
- **Spatial gradients** ($\frac{\partial I}{\partial x}, \frac{\partial I}{\partial y}$) provide boundary and corner information.
- **Temporal gradients** ($\frac{\partial I}{\partial t}$) reveal movement.
- **Spatio-temporal derivatives** ($\frac{\Delta x}{\Delta t}, \frac{\Delta y}{\Delta t}$) yield velocity, heading angle, and acceleration vectors.

This completes the classical Computer Vision pipeline: from raw pixels to preprocessed luminance fields, statistical foreground masks, tracked centroid identities, optical flow fields, and finally interpretable kinematic telemetry.


