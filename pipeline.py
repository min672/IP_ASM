"""
pipeline.py — Unified, importable fingerprint processing pipeline.

Ports the ACTUAL functions from each teammate's notebook (not
reimplemented from scratch) into one module so the GUI can call
run_pipeline(image) and get every stage back in one call, with no
manual PNG hand-off between members.

    Member A -> preprocess_member_a()
    Member B -> process_member_b()
    Member C -> process_member_c()   (simplified: see note in process_member_c)
    Member D -> analyse_member_d()   (minutiae, quality metrics, calibration)

All stages operate on a 256x256 uint8 grayscale image, matching the
team's agreed TARGET_SIZE.
"""

import numpy as np
import cv2
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

TARGET_SIZE = (256, 256)  # (width, height) — team-agreed scale


# =============================================================================
# MEMBER A — Preprocessing + Enhancement
# (ported from Preprocessing_Member_A.ipynb)
# =============================================================================

def resize_image(image, target_size=TARGET_SIZE):
    if image is None or image.size == 0:
        raise ValueError("Cannot resize an empty image.")
    target_width, target_height = target_size
    current_height, current_width = image.shape[:2]
    shrinking = target_width < current_width or target_height < current_height
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC
    return cv2.resize(image, (target_width, target_height), interpolation=interpolation)


def convert_to_grayscale(image):
    if image.ndim == 2:
        gray = image.copy()
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def mean_variance_normalize(image, target_mean=128.0, target_std=50.0):
    image_float = image.astype(np.float32)
    current_mean = float(image_float.mean())
    current_std = float(image_float.std())
    if current_std < 1e-6:
        return image.copy()
    normalized = (image_float - current_mean) / current_std * target_std + target_mean
    return np.clip(normalized, 0, 255).astype(np.uint8)


def contrast_stretch(image):
    minimum, maximum = float(image.min()), float(image.max())
    if maximum <= minimum:
        return image.copy()
    stretched = (image.astype(np.float32) - minimum) * 255.0 / (maximum - minimum)
    return np.clip(stretched, 0, 255).astype(np.uint8)


def preprocess_member_a(
    raw_image,
    target_size=TARGET_SIZE,
    target_mean=128.0,
    target_std=50.0,
    denoising_method="median",
    clahe_clip_limit=2.0,
    clahe_tile_grid=(8, 8),
):
    """Run the full Member A stage on a raw image array (any format)."""
    resized = resize_image(raw_image, target_size)
    grayscale = convert_to_grayscale(resized)
    normalized = mean_variance_normalize(grayscale, target_mean, target_std)

    if denoising_method == "gaussian":
        preprocessed = cv2.GaussianBlur(normalized, (3, 3), sigmaX=0)
    else:
        preprocessed = cv2.medianBlur(normalized, 3)

    contrast_stretched = contrast_stretch(preprocessed)
    histogram_equalized = cv2.equalizeHist(preprocessed)
    clahe_operator = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=clahe_tile_grid)
    clahe_enhanced = clahe_operator.apply(preprocessed)

    return {
        "raw": raw_image,
        "resized": resized,
        "grayscale": grayscale,
        "normalized": normalized,
        "preprocessed": preprocessed,
        "contrast_stretched": contrast_stretched,
        "histogram_equalized": histogram_equalized,
        "clahe": clahe_enhanced,
        "member_b_input": clahe_enhanced.copy(),
    }


# =============================================================================
# MEMBER B — ROI, Orientation, Frequency, Gabor Ridge Recovery
# (ported from Member_B_Ridge_Recovery_Orientation_Analysis.ipynb)
# =============================================================================

def _largest_central_component(mask, minimum_area=300):
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    if count <= 1:
        return np.zeros_like(mask)
    height, width = mask.shape
    centre = np.array([width / 2.0, height / 2.0])
    best_label, best_score = None, -np.inf
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < minimum_area:
            continue
        distance = np.linalg.norm((centroids[label] - centre) / np.array([width, height]))
        score = area * max(0.20, 1.0 - distance)
        if score > best_score:
            best_score, best_label = score, label
    if best_label is None:
        return np.zeros_like(mask)
    result = np.zeros_like(mask)
    result[labels == best_label] = 255
    return result


def _fill_holes(mask):
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flood = padded.copy()
    flood_mask = np.zeros((flood.shape[0] + 2, flood.shape[1] + 2), np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


def extract_roi(image, window_size=15, percentile=50.0):
    image_float = image.astype(np.float32)
    local_mean = cv2.boxFilter(image_float, cv2.CV_32F, (window_size, window_size))
    local_sq_mean = cv2.boxFilter(image_float ** 2, cv2.CV_32F, (window_size, window_size))
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean ** 2, 0))

    margin = max(12, window_size // 2)
    inner = local_std[margin:-margin, margin:-margin]
    threshold = float(np.percentile(inner[inner > 0], percentile))

    dark_threshold = min(220.0, float(np.percentile(image, 70)))
    dark_pixels = (image_float < dark_threshold).astype(np.float32)
    dark_fraction = cv2.boxFilter(dark_pixels, cv2.CV_32F, (window_size, window_size))

    texture_mask = ((local_std >= threshold) & (dark_fraction >= 0.08)).astype(np.uint8) * 255
    texture_mask[:margin, :] = 0
    texture_mask[-margin:, :] = 0
    texture_mask[:, :margin] = 0
    texture_mask[:, -margin:] = 0

    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(texture_mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)
    mask = _largest_central_component(mask)
    mask = _fill_holes(mask)

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=1)
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        hull = cv2.convexHull(max(contours, key=cv2.contourArea))
        mask = np.zeros_like(mask)
        cv2.drawContours(mask, [hull], -1, 255, thickness=cv2.FILLED)

    mask[:margin, :] = 0
    mask[-margin:, :] = 0
    mask[:, :margin] = 0
    mask[:, -margin:] = 0

    coverage = float(np.mean(mask > 0))
    if coverage < 0.08 or coverage > 0.90:
        raise ValueError(f"ROI extraction produced implausible coverage: {coverage:.1%}")

    display = cv2.normalize(local_std, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return mask, local_std, display, threshold


def estimate_orientation(image, roi_mask, sigma=4.0):
    image_float = image.astype(np.float32) / 255.0
    gradient_x = cv2.Sobel(image_float, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(image_float, cv2.CV_32F, 0, 1, ksize=3)
    tensor_xx = cv2.GaussianBlur(gradient_x * gradient_x, (0, 0), sigma)
    tensor_yy = cv2.GaussianBlur(gradient_y * gradient_y, (0, 0), sigma)
    tensor_xy = cv2.GaussianBlur(gradient_x * gradient_y, (0, 0), sigma)
    ridge_orientation = 0.5 * np.arctan2(2.0 * tensor_xy, tensor_xx - tensor_yy) + np.pi / 2.0
    ridge_orientation = np.mod(ridge_orientation, np.pi).astype(np.float32)
    coherence = np.sqrt((tensor_xx - tensor_yy) ** 2 + 4.0 * tensor_xy ** 2) / (tensor_xx + tensor_yy + 1e-8)
    coherence = np.clip(coherence, 0, 1).astype(np.float32)
    ridge_orientation[roi_mask == 0] = 0
    coherence[roi_mask == 0] = 0
    return ridge_orientation, coherence


def orientation_overlay(image, orientation, coherence, roi_mask, step=16, line_length=12, minimum_coherence=0.25):
    overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    half = line_length / 2.0
    for y in range(step // 2, image.shape[0], step):
        for x in range(step // 2, image.shape[1], step):
            if roi_mask[y, x] == 0 or coherence[y, x] < minimum_coherence:
                continue
            theta = float(orientation[y, x])
            dx, dy = half * np.cos(theta), half * np.sin(theta)
            start = (int(round(x - dx)), int(round(y - dy)))
            end = (int(round(x + dx)), int(round(y + dy)))
            cv2.line(overlay, start, end, (255, 0, 0), 1, cv2.LINE_AA)
    return overlay


def _circular_orientation_mean(theta, weights):
    sine = np.sum(weights * np.sin(2.0 * theta))
    cosine = np.sum(weights * np.cos(2.0 * theta))
    return float(np.mod(0.5 * np.arctan2(sine, cosine), np.pi))


def _estimate_block_frequency(block, ridge_orientation, minimum_wavelength, maximum_wavelength):
    block_size = block.shape[0]
    rotation_degrees = 90.0 - np.degrees(ridge_orientation)
    centre = ((block_size - 1) / 2.0, (block_size - 1) / 2.0)
    matrix = cv2.getRotationMatrix2D(centre, rotation_degrees, 1.0)
    rotated = cv2.warpAffine(block.astype(np.float32), matrix, (block_size, block_size),
                              flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
    profile = np.mean(rotated, axis=0)
    profile = profile - cv2.GaussianBlur(profile.reshape(1, -1), (0, 0), 2.0).ravel()
    profile = profile - np.mean(profile)
    if np.std(profile) < 1e-6:
        return 0.0, 0.0
    padded_length = 256
    spectrum = np.abs(np.fft.rfft(profile * np.hanning(profile.size), n=padded_length)) ** 2
    frequencies = np.fft.rfftfreq(padded_length)
    valid = (frequencies >= 1.0 / maximum_wavelength) & (frequencies <= 1.0 / minimum_wavelength)
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        return 0.0, 0.0
    band_power = spectrum[valid_indices]
    peak_index = valid_indices[int(np.argmax(band_power))]
    peak_frequency = float(frequencies[peak_index])
    confidence = float(spectrum[peak_index] / (np.sum(band_power) + 1e-8))
    if confidence < 0.08:
        return 0.0, confidence
    return peak_frequency, confidence


def estimate_frequency(image, orientation, coherence, roi_mask, block_size=32, step=16,
                        minimum_wavelength=6.0, maximum_wavelength=16.0):
    height, width = image.shape
    frequency_sum = np.zeros_like(image, dtype=np.float32)
    frequency_weight = np.zeros_like(image, dtype=np.float32)
    estimates = []
    for y in range(0, height - block_size + 1, step):
        for x in range(0, width - block_size + 1, step):
            region = np.s_[y:y + block_size, x:x + block_size]
            valid = roi_mask[region] > 0
            if np.mean(valid) < 0.60:
                continue
            mean_coherence = float(np.mean(coherence[region][valid]))
            if mean_coherence < 0.20:
                continue
            theta = _circular_orientation_mean(orientation[region][valid], coherence[region][valid])
            frequency, confidence = _estimate_block_frequency(image[region], theta, minimum_wavelength, maximum_wavelength)
            if frequency <= 0:
                continue
            weight = confidence * mean_coherence
            frequency_sum[region] += frequency * weight
            frequency_weight[region] += weight
            estimates.append(frequency)
    if not estimates:
        raise ValueError("Frequency estimation failed: no valid fingerprint blocks were found.")
    median_frequency = float(np.median(estimates))
    local_frequency = np.full_like(image, median_frequency, dtype=np.float32)
    known = frequency_weight > 1e-8
    local_frequency[known] = frequency_sum[known] / frequency_weight[known]
    local_frequency[roi_mask == 0] = 0
    return local_frequency, np.asarray(estimates, dtype=np.float32), median_frequency


def gabor_recovery(image, orientation, coherence, roi_mask, wavelength,
                    number_of_orientations=16, kernel_size=25, minimum_coherence=0.20):
    image_float = image.astype(np.float32)
    local_mean = cv2.GaussianBlur(image_float, (0, 0), 4.0)
    centred = image_float - local_mean
    orientation_indices = np.floor(np.mod(orientation, np.pi) / np.pi * number_of_orientations).astype(np.int32)
    orientation_indices = np.clip(orientation_indices, 0, number_of_orientations - 1)
    selected_response = np.zeros_like(image_float)
    for index in range(number_of_orientations):
        ridge_theta = index * np.pi / number_of_orientations
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma=4.0, theta=ridge_theta + np.pi / 2.0,
                                     lambd=wavelength, gamma=0.55, psi=0, ktype=cv2.CV_32F)
        kernel -= np.mean(kernel)
        kernel /= np.sum(np.abs(kernel)) + 1e-8
        response = cv2.filter2D(centred, cv2.CV_32F, kernel, borderType=cv2.BORDER_REFLECT)
        choose = (orientation_indices == index) & (roi_mask > 0) & (coherence >= minimum_coherence)
        selected_response[choose] = response[choose]

    reliable = (roi_mask > 0) & (coherence >= minimum_coherence)
    values = selected_response[reliable]
    if values.size == 0:
        raise ValueError("Gabor recovery failed: no reliable orientation pixels were found.")
    lower, upper = np.percentile(values, [2, 98])
    scaled = np.clip((selected_response - lower) * 255.0 / (upper - lower + 1e-8), 0, 255).astype(np.uint8)

    correlation = np.corrcoef(image[reliable].astype(np.float32), scaled[reliable].astype(np.float32))[0, 1]
    if np.isfinite(correlation) and correlation < 0:
        scaled = cv2.bitwise_not(scaled)

    confidence = np.clip((coherence - minimum_coherence) / (1.0 - minimum_coherence), 0, 1)
    gabor_weight = 0.50 * confidence
    recovered = (1.0 - gabor_weight) * image_float + gabor_weight * scaled.astype(np.float32)
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    recovered[roi_mask == 0] = 255
    scaled[roi_mask == 0] = 255
    return recovered, scaled, selected_response


def process_member_b(enhanced_input,
                      roi_window_size=15, roi_texture_percentile=50.0,
                      orientation_sigma=4.0, minimum_coherence=0.20,
                      freq_block_size=32, freq_step=16,
                      min_wavelength=6.0, max_wavelength=16.0,
                      n_gabor_orientations=16, gabor_kernel_size=25):
    """Run the full Member B stage on Member A's CLAHE output."""
    roi_mask, local_texture_strength, roi_texture_display, roi_threshold = extract_roi(
        enhanced_input, window_size=roi_window_size, percentile=roi_texture_percentile)

    orientation_map, orientation_coherence = estimate_orientation(
        enhanced_input, roi_mask, sigma=orientation_sigma)

    orientation_visualisation = orientation_overlay(
        enhanced_input, orientation_map, orientation_coherence, roi_mask,
        step=16, line_length=12, minimum_coherence=minimum_coherence)

    frequency_map, valid_frequency_estimates, median_ridge_frequency = estimate_frequency(
        enhanced_input, orientation_map, orientation_coherence, roi_mask,
        block_size=freq_block_size, step=freq_step,
        minimum_wavelength=min_wavelength, maximum_wavelength=max_wavelength)

    estimated_ridge_wavelength = 1.0 / median_ridge_frequency

    member_b_output, gabor_enhanced, _ = gabor_recovery(
        enhanced_input, orientation_map, orientation_coherence, roi_mask,
        wavelength=estimated_ridge_wavelength,
        number_of_orientations=n_gabor_orientations,
        kernel_size=gabor_kernel_size,
        minimum_coherence=minimum_coherence)

    return {
        "roi_mask": roi_mask,
        "roi_texture_display": roi_texture_display,
        "orientation_map": orientation_map,
        "orientation_coherence": orientation_coherence,
        "orientation_visualisation": orientation_visualisation,
        "frequency_map": frequency_map,
        "median_ridge_frequency": median_ridge_frequency,
        "estimated_ridge_wavelength": estimated_ridge_wavelength,
        "gabor_enhanced": gabor_enhanced,
        "member_b_output": member_b_output,
        "member_c_input": member_b_output.copy(),
    }


# =============================================================================
# MEMBER C — Adaptive Threshold, Morphology, Skeletonization
# (ported from Member_C (third).ipynb, core algorithm only — see module
#  docstring: the notebook's local/automatic morphology search and its own
#  separate GrabCut ROI extraction are not replicated here for GUI
#  performance; this uses Member B's roi_mask and Member C's manually
#  selected final method: adaptive Gaussian threshold + opening->closing.)
# =============================================================================

def apply_adaptive_gaussian_threshold(image, block_size=13, C=5):
    if block_size % 2 == 0:
        block_size += 1
    return cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)


def apply_opening(image, kernel):
    foreground = cv2.bitwise_not(image)
    output = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, kernel)
    return cv2.bitwise_not(output)


def apply_closing(image, kernel):
    foreground = cv2.bitwise_not(image)
    output = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel)
    return cv2.bitwise_not(output)


def zhang_suen_thinning(image):
    """Verbatim port of Member C's Zhang-Suen thinning algorithm."""
    image = (image // 255).astype(np.uint8)
    changing = True
    while changing:
        changing = False
        rows, cols = image.shape

        remove_pixels = []
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if image[y, x] != 1:
                    continue
                P2, P3, P4 = image[y-1, x], image[y-1, x+1], image[y, x+1]
                P5, P6, P7 = image[y+1, x+1], image[y+1, x], image[y+1, x-1]
                P8, P9 = image[y, x-1], image[y-1, x-1]
                neighbours = [P2, P3, P4, P5, P6, P7, P8, P9]
                B = sum(neighbours)
                A = sum(neighbours[i] == 0 and neighbours[(i + 1) % 8] == 1 for i in range(8))
                if 2 <= B <= 6 and A == 1 and P2 * P4 * P6 == 0 and P4 * P6 * P8 == 0:
                    remove_pixels.append((y, x))
        if remove_pixels:
            changing = True
            for y, x in remove_pixels:
                image[y, x] = 0

        remove_pixels = []
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                if image[y, x] != 1:
                    continue
                P2, P3, P4 = image[y-1, x], image[y-1, x+1], image[y, x+1]
                P5, P6, P7 = image[y+1, x+1], image[y+1, x], image[y+1, x-1]
                P8, P9 = image[y, x-1], image[y-1, x-1]
                neighbours = [P2, P3, P4, P5, P6, P7, P8, P9]
                B = sum(neighbours)
                A = sum(neighbours[i] == 0 and neighbours[(i + 1) % 8] == 1 for i in range(8))
                if 2 <= B <= 6 and A == 1 and P2 * P4 * P8 == 0 and P2 * P6 * P8 == 0:
                    remove_pixels.append((y, x))
        if remove_pixels:
            changing = True
            for y, x in remove_pixels:
                image[y, x] = 0

    return (image * 255).astype(np.uint8)


def process_member_c(member_c_input, roi_mask, adaptive_block_size=13, adaptive_C=5, morph_kernel_size=3):
    """Run the (simplified) Member C stage: threshold -> morphology -> skeleton."""
    masked_input = np.where(roi_mask > 0, member_c_input, 255).astype(np.uint8)
    smoothed = cv2.medianBlur(masked_input, 5)

    adaptive_threshold = apply_adaptive_gaussian_threshold(smoothed, block_size=adaptive_block_size, C=adaptive_C)
    adaptive_threshold = np.where(roi_mask > 0, adaptive_threshold, 255).astype(np.uint8)

    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    opened = apply_opening(adaptive_threshold, kernel)
    morphology_output = apply_closing(opened, kernel)

    skeleton_255 = zhang_suen_thinning(morphology_output)
    skeleton = (skeleton_255 > 127).astype(np.uint8)

    return {
        "adaptive_threshold": adaptive_threshold,
        "morphology_output": morphology_output,
        "skeleton_255": skeleton_255,
        "skeleton": skeleton,
        "member_d_input": skeleton_255.copy(),
    }


# =============================================================================
# MEMBER D — Minutiae Detection, False Minutiae Filtering, Quality Metrics
# =============================================================================

_NEIGHBOUR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def crossing_number(skeleton, y, x):
    vals = [skeleton[y + dy, x + dx] for dy, dx in _NEIGHBOUR_OFFSETS]
    vals.append(vals[0])
    return sum(abs(int(vals[i]) - int(vals[i + 1])) for i in range(8)) // 2


def extract_minutiae(skeleton):
    h, w = skeleton.shape
    minutiae = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if skeleton[y, x] != 1:
                continue
            cn = crossing_number(skeleton, y, x)
            if cn == 1:
                minutiae.append({"x": x, "y": y, "type": "ending", "cn": cn})
            elif cn >= 3:
                minutiae.append({"x": x, "y": y, "type": "bifurcation", "cn": cn})
    return minutiae


def trace_ridge_length(skeleton, y, x, max_len=20):
    visited = {(y, x)}
    current = (y, x)
    steps = 0
    while steps < max_len:
        cy, cx = current
        next_pixel = None
        for dy, dx in _NEIGHBOUR_OFFSETS:
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]
                    and skeleton[ny, nx] == 1 and (ny, nx) not in visited):
                next_pixel = (ny, nx)
                break
        if next_pixel is None:
            break
        visited.add(next_pixel)
        current = next_pixel
        steps += 1
    return steps


def trace_ridge_path(skeleton, y, x, max_len=30):
    """Like trace_ridge_length, but records the actual path coordinates
    so straightness can be evaluated."""
    visited = {(y, x)}
    path = [(y, x)]
    current = (y, x)
    steps = 0
    while steps < max_len:
        cy, cx = current
        next_pixel = None
        for dy, dx in _NEIGHBOUR_OFFSETS:
            ny, nx = cy + dy, cx + dx
            if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]
                    and skeleton[ny, nx] == 1 and (ny, nx) not in visited):
                next_pixel = (ny, nx)
                break
        if next_pixel is None:
            break
        visited.add(next_pixel)
        path.append(next_pixel)
        current = next_pixel
        steps += 1
    return path


def straightness_score(path):
    """
    1.0 = perfectly straight, lower = more curved. Uses actual cumulative
    path length (accounts for diagonal steps) as the denominator, not
    step count, so it is not biased by movement direction.
    """
    if len(path) < 2:
        return 0.0
    (y0, x0), (y1, x1) = path[0], path[-1]
    straight_dist = np.hypot(y1 - y0, x1 - x0)
    walked_dist = sum(
        np.hypot(path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    )
    return straight_dist / walked_dist if walked_dist > 0 else 0.0


def filter_by_straightness(minutiae, skeleton, max_straightness=0.97, trace_len=30):
    """
    Reject minutiae sitting on an abnormally straight ridge segment.
    Real fingerprint ridges curve with the natural shape of the finger -
    a segment that stays almost perfectly straight for a long trace is a
    strong signal of a geometric artefact (e.g. a convex-hull ROI edge,
    a cut/obliteration boundary, or any other non-ridge structure) -
    this works purely on the skeleton itself, with no dependency on
    Member A/B/C's intermediate data.
    """
    survivors = []
    for m in minutiae:
        path = trace_ridge_path(skeleton, m["y"], m["x"], max_len=trace_len)
        if straightness_score(path) < max_straightness:
            survivors.append(m)
    return survivors


def filter_false_minutiae(minutiae, skeleton, roi_mask=None,
                           min_ridge_length=6, min_pair_distance=10, boundary_margin=10):
    h, w = skeleton.shape

    def near_boundary(m):
        if roi_mask is not None:
            roi_bool = roi_mask > 0
            dist = ndimage.distance_transform_edt(roi_bool)
            return dist[m["y"], m["x"]] < boundary_margin
        return (m["x"] < boundary_margin or m["x"] > w - boundary_margin or
                m["y"] < boundary_margin or m["y"] > h - boundary_margin)

    survivors = [m for m in minutiae if not near_boundary(m)]

    checked = []
    for m in survivors:
        length = trace_ridge_length(skeleton, m["y"], m["x"], max_len=min_ridge_length * 2)
        if length >= min_ridge_length:
            checked.append(m)
    survivors = checked

    to_remove = set()
    for i in range(len(survivors)):
        for j in range(i + 1, len(survivors)):
            a, b = survivors[i], survivors[j]
            dist = np.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if dist < min_pair_distance:
                to_remove.add(i)
                to_remove.add(j)

    return [m for idx, m in enumerate(survivors) if idx not in to_remove]


def filter_by_local_density(minutiae, max_neighbours=5, neighbour_radius=15):
    """
    Reject minutiae sitting in an abnormally dense local cluster - a real
    fingerprint has minutiae spread out, not packed into a small area.
    A tight mesh/web artefact (from thresholding noise in a blurred
    region) typically produces many minutiae within a small radius of
    each other, which this rule catches directly.
    """
    survivors = []
    for i, m in enumerate(minutiae):
        neighbour_count = 0
        for j, other in enumerate(minutiae):
            if i == j:
                continue
            dist = np.hypot(m["x"] - other["x"], m["y"] - other["y"])
            if dist <= neighbour_radius:
                neighbour_count += 1
        if neighbour_count <= max_neighbours:
            survivors.append(m)
    return survivors


def detect_true_minutiae(skeleton, roi_mask=None, max_neighbours=5, neighbour_radius=15,
                          max_straightness=0.97, straightness_trace_len=30, **filter_kwargs):
    raw = extract_minutiae(skeleton)
    true_minutiae = filter_false_minutiae(raw, skeleton, roi_mask=roi_mask, **filter_kwargs)
    true_minutiae = filter_by_local_density(true_minutiae, max_neighbours=max_neighbours,
                                             neighbour_radius=neighbour_radius)
    true_minutiae = filter_by_straightness(true_minutiae, skeleton,
                                            max_straightness=max_straightness,
                                            trace_len=straightness_trace_len)
    return {
        "raw_count": len(raw),
        "true_count": len(true_minutiae),
        "removed_count": len(raw) - len(true_minutiae),
        "minutiae": true_minutiae,
    }


def compute_ridge_density(skeleton, sample_lines=20, line_length=20, seed=42):
    h, w = skeleton.shape
    rng = np.random.default_rng(seed)
    densities = []
    for _ in range(sample_lines):
        y = rng.integers(0, h)
        x_start = rng.integers(0, max(1, w - line_length))
        line = skeleton[y, x_start:x_start + line_length]
        crossings = np.sum((line[:-1] == 0) & (line[1:] == 1))
        densities.append(crossings / line_length)
    return float(np.mean(densities))


def compute_quality_score(ridge_density, true_minutiae_count,
                           w_density=0.5, w_minutiae=0.5, expected_max_minutiae=60):
    """Simplified NFIQ2-inspired proxy score (0-100). Not the official NFIQ2 algorithm."""
    norm_density = min(ridge_density / 0.5, 1.0)
    norm_minutiae = min(true_minutiae_count / expected_max_minutiae, 1.0)
    return round((w_density * norm_density + w_minutiae * norm_minutiae) * 100, 2)


def evaluate_enhancement(image_a, image_b):
    """SSIM and PSNR between two same-shape grayscale images."""
    if image_a.shape != image_b.shape:
        raise ValueError(f"Shape mismatch: {image_a.shape} vs {image_b.shape}.")
    return {
        "SSIM": round(float(ssim(image_a, image_b, data_range=255)), 4),
        "PSNR": round(float(psnr(image_a, image_b, data_range=255)), 2),
    }


# --- Image Calibration ------------------------------------------------------
ORIGINAL_DPI = 500                 # SOCOFing capture spec (Shehu et al., 2018)
ORIGINAL_SIZE_PX = (96, 103)       # SOCOFing native scan size (width, height)


def compute_calibration(working_size_px, original_size_px=ORIGINAL_SIZE_PX, original_dpi=ORIGINAL_DPI):
    scale_x = working_size_px[0] / original_size_px[0]
    scale_y = working_size_px[1] / original_size_px[1]
    mm_per_px_original = 25.4 / original_dpi
    return {
        "scale_x": scale_x, "scale_y": scale_y,
        "mm_per_px_x": mm_per_px_original / scale_x,
        "mm_per_px_y": mm_per_px_original / scale_y,
    }


def build_reliable_mask(roi_mask, orientation_coherence, min_coherence=0.25):
    """
    Regions where Member B's orientation estimate was low-confidence are
    unreliable for everything downstream (threshold, skeleton, minutiae).
    Excluding them directly addresses the straight-line / mesh artefacts
    that appear where adaptive thresholding was applied blindly to
    low-coherence (near-background or blurred) areas.
    """
    reliable = (roi_mask > 0) & (orientation_coherence >= min_coherence)
    return (reliable.astype(np.uint8)) * 255


def analyse_member_d(skeleton, roi_mask, raw_gray_resized, enhanced_image,
                      orientation_coherence=None, min_coherence=0.25,
                      min_ridge_length=6, min_pair_distance=10, boundary_margin=10,
                      min_usable_minutiae=12):
    """Full Member D stage: Part 1 (enhancement effectiveness) + Part 2 (matching
    suitability) + Image Calibration, given the full set of pipeline images."""
    # Restrict analysis to regions Member B was actually confident about,
    # not just "inside the ROI" - this filters out the straight-line and
    # mesh artefacts that come from low-coherence regions being
    # thresholded/skeletonized anyway.
    if orientation_coherence is not None:
        effective_mask = build_reliable_mask(roi_mask, orientation_coherence, min_coherence)
    else:
        effective_mask = roi_mask

    raw_minutiae = extract_minutiae(skeleton)
    detection_result = detect_true_minutiae(
        skeleton, roi_mask=effective_mask,
        min_ridge_length=min_ridge_length,
        min_pair_distance=min_pair_distance,
        boundary_margin=boundary_margin,
    )
    true_minutiae = detection_result["minutiae"]
    true_coords = {(m["x"], m["y"]) for m in true_minutiae}
    false_minutiae = [m for m in raw_minutiae if (m["x"], m["y"]) not in true_coords]

    ending_count = sum(1 for m in true_minutiae if m["type"] == "ending")
    bifurcation_count = sum(1 for m in true_minutiae if m["type"] == "bifurcation")

    ridge_density = compute_ridge_density(skeleton)
    quality_score = compute_quality_score(ridge_density, detection_result["true_count"])
    suitable_for_matching = detection_result["true_count"] >= min_usable_minutiae

    enhancement_metrics = evaluate_enhancement(raw_gray_resized, enhanced_image)

    calibration = compute_calibration(working_size_px=(skeleton.shape[1], skeleton.shape[0]))
    avg_ridge_spacing_px = 1 / ridge_density if ridge_density > 0 else float("nan")
    avg_ridge_spacing_mm = avg_ridge_spacing_px * calibration["mm_per_px_x"]
    image_width_mm = skeleton.shape[1] * calibration["mm_per_px_x"]
    image_height_mm = skeleton.shape[0] * calibration["mm_per_px_y"]

    return {
        "true_minutiae_list": true_minutiae,
        "false_minutiae_list": false_minutiae,
        "raw_minutiae": detection_result["raw_count"],
        "true_minutiae": detection_result["true_count"],
        "false_minutiae_removed": detection_result["removed_count"],
        "ridge_endings": ending_count,
        "bifurcations": bifurcation_count,
        "ridge_density": round(ridge_density, 4),
        "quality_score": quality_score,
        "suitable_for_matching": suitable_for_matching,
        "SSIM_raw_vs_enhanced": enhancement_metrics["SSIM"],
        "PSNR_raw_vs_enhanced_dB": enhancement_metrics["PSNR"],
        "mm_per_px": round(calibration["mm_per_px_x"], 4),
        "avg_ridge_spacing_mm": round(avg_ridge_spacing_mm, 3),
        "image_size_mm": f"{image_width_mm:.1f} x {image_height_mm:.1f}",
    }


# =============================================================================
# FULL PIPELINE — single entry point the GUI calls
# =============================================================================

def run_pipeline(raw_image, member_a_kwargs=None, member_b_kwargs=None,
                  member_c_kwargs=None, member_d_kwargs=None):
    """
    Run the ENTIRE pipeline on one raw image array (as read by cv2.imread).
    Returns a dict with every intermediate stage plus the final Member D
    analysis, ready for a GUI to display or a batch runner to aggregate.
    """
    a_kwargs = member_a_kwargs or {}
    b_kwargs = member_b_kwargs or {}
    c_kwargs = member_c_kwargs or {}
    d_kwargs = member_d_kwargs or {}

    a = preprocess_member_a(raw_image, **a_kwargs)
    b = process_member_b(a["member_b_input"], **b_kwargs)
    c = process_member_c(b["member_c_input"], b["roi_mask"], **c_kwargs)

    raw_gray_resized = resize_image(convert_to_grayscale(raw_image), TARGET_SIZE)

    d = analyse_member_d(
        skeleton=c["skeleton"],
        roi_mask=b["roi_mask"],
        raw_gray_resized=raw_gray_resized,
        enhanced_image=a["member_b_input"],
        orientation_coherence=b["orientation_coherence"],
        **d_kwargs,
    )

    return {
        "raw_gray_resized": raw_gray_resized,
        **a,
        **b,
        **c,
        "member_d": d,
    }
