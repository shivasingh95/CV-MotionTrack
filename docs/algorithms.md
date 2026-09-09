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
