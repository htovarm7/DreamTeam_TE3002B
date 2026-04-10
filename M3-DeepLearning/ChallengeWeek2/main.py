"""
Reto: Clasificación de Flechas de Señalización Vial con Regresión Logística
TE3002B — Prof. Alberto Munoz — Tecnologico de Monterrey

Dataset From: https://universe.roboflow.com/ntu-sbehf/task_2_image_recognition

Approach:
  - Synthetic dataset generated with OpenCV (diverse arrow styles)
  - Logistic Regression implemented from scratch (gradient descent)
  - Feature: horizontal column projections + edge asymmetry
  - Evaluation: accuracy, precision, recall, F1, confusion matrix

Usage:
  python main.py                  # train + evaluate (default)
  python main.py --camera         # train then open live camera inference
  python main.py --dataset        # train then browse real dataset images
  python main.py --image path.jpg # predict on a single image file
"""

import argparse
import cv2
import numpy as np
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from sklearn.linear_model import LogisticRegression as SKLearnLR

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')

IMG_SIZE = 64 
N_PER_CLASS = 600

ARROW_STYLES = ['filled', 'outline', 'chevron', 'line', 'double', 'curved']
COLORS = [
    (255, 255, 255), (0, 0, 0), (200, 50, 50),
    (50, 200, 50),   (50, 50, 200), (255, 200, 0),
    (150, 0, 150),   (0, 200, 200),
]
BG_COLORS = [
    (0,   0,   0), (255, 255, 255), (50,  50,  50),
    (200, 200, 200), (30, 60, 30),  (60, 30, 30),
    (220, 220, 180),
]


def random_color(exclude=None):
    c = random.choice(COLORS)
    if exclude and c == exclude:
        c = random.choice(COLORS)
    return c


def draw_filled_arrow(img, cx, cy, size, direction, color, thickness=None):
    """Standard filled polygon arrow. thickness ignored (polygon fill)."""
    _ = thickness
    half = size // 2
    body_w = max(int(size * 0.22), 2)
    pts = np.array([
        [cx + half,   cy],
        [cx,          cy - half],
        [cx,          cy - body_w],
        [cx - half,   cy - body_w],
        [cx - half,   cy + body_w],
        [cx,          cy + body_w],
        [cx,          cy + half],
    ], dtype=np.int32)
    if direction == 0:   # left — mirror horizontally
        pts[:, 0] = 2 * cx - pts[:, 0]
    cv2.fillPoly(img, [pts], color)


def draw_outline_arrow(img, cx, cy, size, direction, color, thickness=None):
    """Same shape but drawn as polylines."""
    half = size // 2
    body_w = max(int(size * 0.22), 2)
    pts = np.array([
        [cx + half, cy],
        [cx, cy - half],
        [cx, cy - body_w],
        [cx - half, cy - body_w],
        [cx - half, cy + body_w],
        [cx, cy + body_w],
        [cx, cy + half],
    ], dtype=np.int32)
    if direction == 0:
        pts[:, 0] = 2 * cx - pts[:, 0]
    t = thickness if thickness else random.randint(2, 5)
    cv2.polylines(img, [pts], isClosed=True, color=color, thickness=t)


def draw_chevron_arrow(img, cx, cy, size, direction, color, thickness=None):
    """Chevron / V-shaped arrow using lines."""
    t = thickness if thickness else random.randint(3, 8)
    half = size // 2
    tip_x = cx + half if direction == 1 else cx - half
    pts = np.array([
        [cx - half if direction == 1 else cx + half, cy - half],
        [tip_x, cy],
        [cx - half if direction == 1 else cx + half, cy + half],
    ], dtype=np.int32)
    cv2.polylines(img, [pts], isClosed=False, color=color, thickness=t)


def draw_line_arrow(img, cx, cy, size, direction, color, thickness=None):
    """cv2.arrowedLine style."""
    t = thickness if thickness else random.randint(2, 6)
    half = size // 2
    tip_length = random.uniform(0.25, 0.45)
    if direction == 1:
        p1, p2 = (cx - half, cy), (cx + half, cy)
    else:
        p1, p2 = (cx + half, cy), (cx - half, cy)
    cv2.arrowedLine(img, p1, p2, color, t, tipLength=tip_length)


def draw_double_arrow(img, cx, cy, size, direction, color, thickness=None):
    """Two concentric arrows. thickness ignored (delegated)."""
    _ = thickness
    draw_filled_arrow(img, cx, cy, size, direction, color)
    inner_size = max(int(size * 0.55), 10)
    draw_outline_arrow(img, cx, cy, inner_size, direction, color)


def draw_curved_arrow(img, cx, cy, size, direction, color, thickness=None):
    """Curved arc arrow using ellipse + line tip."""
    t = thickness if thickness else random.randint(2, 5)
    axes = (size // 2, size // 3)
    start_angle = 200 if direction == 1 else 340
    end_angle   = 340 if direction == 1 else 200
    cv2.ellipse(img, (cx, cy), axes, 0, start_angle, end_angle, color, t)
    tip_x = cx + size // 2 if direction == 1 else cx - size // 2
    tip_y = cy
    cv2.arrowedLine(img, (tip_x, tip_y - 8), (tip_x, tip_y + 8),
                    color, t, tipLength=0.5)


DRAW_FN = {
    'filled':  draw_filled_arrow,
    'outline': draw_outline_arrow,
    'chevron': draw_chevron_arrow,
    'line':    draw_line_arrow,
    'double':  draw_double_arrow,
    'curved':  draw_curved_arrow,
}


def add_noise(img, level=0.05):
    """Additive Gaussian noise — simulates sensor noise / JPEG compression."""
    noise = (np.random.randn(*img.shape) * 255 * level).astype(np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def add_lighting(img):
    """
    Simulate non-uniform illumination / shadow:
    a linear gradient darkens one side of the image, mimicking
    directional lighting found on real road signs.
    """
    h, w  = img.shape[:2]
    alpha = random.uniform(0.3, 0.7)
    side  = random.choice(['left', 'right', 'top', 'bottom'])
    if side == 'left':
        grad = np.tile(np.linspace(0, 1, w), (h, 1))
    elif side == 'right':
        grad = np.tile(np.linspace(1, 0, w), (h, 1))
    elif side == 'top':
        grad = np.tile(np.linspace(0, 1, h), (w, 1)).T
    else:
        grad = np.tile(np.linspace(1, 0, h), (w, 1)).T
    mask = (alpha + (1 - alpha) * grad.astype(np.float32))[..., np.newaxis]
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def generate_arrow_image(direction, canvas=128):
    """
    Generate one arrow image.
    direction: 0=left, 1=right
    """
    bg = random.choice(BG_COLORS)
    img = np.full((canvas, canvas, 3), bg, dtype=np.uint8)

    style = random.choice(ARROW_STYLES)
    fg = random_color(exclude=bg)

    size = random.randint(int(canvas * 0.35), int(canvas * 0.70))
    cx = canvas // 2 + random.randint(-10, 10)
    cy = canvas // 2 + random.randint(-10, 10)

    DRAW_FN[style](img, cx, cy, size, direction, fg)

    # Small rotation augmentation
    angle = random.uniform(-20, 20)
    M = cv2.getRotationMatrix2D((canvas / 2, canvas / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (canvas, canvas),
                         borderValue=bg)

    # Occasional noise
    if random.random() < 0.4:
        img = add_noise(img, level=random.uniform(0.02, 0.08))

    # Occasional Gaussian blur
    if random.random() < 0.3:
        k = random.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Lighting / shadow gradient
    if random.random() < 0.35:
        img = add_lighting(img)

    return img


def build_dataset(n_per_class=N_PER_CLASS, canvas=128, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    images, labels = [], []
    for direction in [0, 1]:
        for _ in range(n_per_class):
            img = generate_arrow_image(direction, canvas)
            images.append(img)
            labels.append(direction)
    return images, labels

def center_by_bbox(gray):
    """
    Alignment / centering: locate the arrow via Otsu threshold + contours,
    crop to its bounding box with padding. Handles dark-on-light and
    light-on-dark images.
    """
    _, bw = cv2.threshold(gray, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    best_bbox, best_area = None, 0
    for cand in [bw, cv2.bitwise_not(bw)]:
        contours, _ = cv2.findContours(cand, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        big = [c for c in contours if cv2.contourArea(c) > 10]
        if not big:
            continue
        x, y, w, h = cv2.boundingRect(np.vstack(big))
        if w * h > best_area:
            best_area = w * h
            best_bbox = (x, y, w, h)
    if best_bbox is None or best_area < 50:
        return gray
    x, y, w, h = best_bbox
    pad = max(5, int(max(w, h) * 0.10))
    H, W = gray.shape
    return gray[max(0, y - pad):min(H, y + h + pad),
                max(0, x - pad):min(W, x + w + pad)]


def preprocess(img, size=IMG_SIZE):
    """
    Full preprocessing pipeline (rubric steps):
      1. Grayscale conversion
      2. Centering by bounding box (alignment)
      3. Resize to IMG_SIZE × IMG_SIZE
      4. Normalize to [0,1]          → gray_norm
      5. Otsu binarization           → binary_norm  (binarización)
      6. Canny edge extraction       → edges
    Returns (gray_norm, binary_norm, edges) — float32, shape (size, size).
    """
    gray        = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cropped     = center_by_bbox(gray)
    resized     = cv2.resize(cropped, (size, size))
    gray_norm   = resized.astype(np.float32) / 255.0
    _, binary   = cv2.threshold(resized, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary_norm = binary.astype(np.float32) / 255.0
    edges       = cv2.Canny(resized, 50, 150).astype(np.float32) / 255.0
    return gray_norm, binary_norm, edges


def _norm_proj(arr):
    s = arr.sum()
    return arr / (s + 1e-8)


def extract_features(gray_norm, binary_norm, edges):
    """
    Feature vector — each block justified by left/right asymmetry:

    A) Column projections (horizontal sums) — raw, binary, edge
       WHY: a right arrow has heavier pixel mass toward the right columns;
            the profile shifts toward the arrowhead side.

    B) Row projections (vertical sums) — raw, edge
       WHY: arrowhead vs. tail create different vertical distributions;
            adds orthogonal shape information.

    C) Pixel-intensity histogram (16 bins)
       WHY: captures contrast distribution; normalizes for brightness
            while preserving which half of the image is denser.

    D) Left–right mass asymmetry scalars — raw, binary, edge
       WHY: strongest direct signal: for a right arrow right_mass > left_mass.

    E) Hu moments of binary map (7, log-scaled)
       WHY: rotation-invariant shape descriptors encoding elongation
            and compactness of the arrowhead region.

    F) Centroid X (relative, in [-1, 1])
       WHY: the arrowhead mass shifts the centroid toward the tip side.

    Total: 5×64 + 16 + 3 + 7 + 1 = 347 dimensions.
    """
    h = IMG_SIZE // 2

    # A. Column projections
    col_raw  = _norm_proj(gray_norm.sum(axis=0))
    col_bin  = _norm_proj(binary_norm.sum(axis=0))
    col_edge = _norm_proj(edges.sum(axis=0))

    # B. Row projections
    row_raw  = _norm_proj(gray_norm.sum(axis=1))
    row_edge = _norm_proj(edges.sum(axis=1))

    # C. Pixel histogram
    hist, _ = np.histogram(gray_norm.ravel(), bins=16, range=(0, 1))
    hist    = hist.astype(np.float32) / (hist.sum() + 1e-8)

    # D. Left–right asymmetry
    def lr(arr):
        l = arr[:, :h].sum()
        r = arr[:, h:].sum()
        return (r - l) / (r + l + 1e-8)

    # E. Hu moments of binary map
    mom = cv2.moments(binary_norm)
    hu  = cv2.HuMoments(mom).flatten()
    hu  = np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    # F. Centroid X
    cx_rel = (mom['m10'] / (mom['m00'] + 1e-8) / IMG_SIZE) * 2 - 1

    return np.concatenate([
        col_raw,                        # 64
        col_bin,                        # 64
        col_edge,                       # 64
        row_raw,                        # 64
        row_edge,                       # 64
        hist,                           # 16
        [lr(gray_norm), lr(binary_norm), lr(edges)],   # 3
        hu,                             # 7
        [cx_rel],                       # 1
    ])                                  # Total = 347


def build_feature_matrix(images):
    feats = []
    for img in images:
        g, b, e = preprocess(img)
        feats.append(extract_features(g, b, e))
    return np.array(feats)


def evaluate(y_true, y_pred, name="Model"):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"         Pred Left  Pred Right")
    print(f"  Left      {cm[0,0]:5d}       {cm[0,1]:5d}")
    print(f"  Right     {cm[1,0]:5d}       {cm[1,1]:5d}")
    print(classification_report(y_true, y_pred,
                                 target_names=['Left(0)', 'Right(1)']))
    return acc, prec, rec, f1, cm

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_sample_images(images, labels, n=16):
    """Save a grid of sample arrows."""
    _, axes = plt.subplots(2, n // 2, figsize=(n * 1.2, 5))
    left_imgs  = [(img, lbl) for img, lbl in zip(images, labels) if lbl == 0][:n//2]
    right_imgs = [(img, lbl) for img, lbl in zip(images, labels) if lbl == 1][:n//2]
    for col, (img, _) in enumerate(left_imgs):
        axes[0, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[0, col].axis('off')
        axes[0, col].set_title('Left', fontsize=8)
    for col, (img, _) in enumerate(right_imgs):
        axes[1, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[1, col].axis('off')
        axes[1, col].set_title('Right', fontsize=8)
    plt.suptitle('Sample Dataset Images', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'sample_images.png'), dpi=100)
    plt.close()
    print("  [saved] sample_images.png")


def plot_training_curve(losses):
    plt.figure(figsize=(7, 4))
    plt.plot(losses, color='steelblue')
    plt.xlabel('Iteration')
    plt.ylabel('Binary Cross-Entropy Loss')
    plt.title('Training Loss Curve (Scratch LR)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'training_curve.png'), dpi=100)
    plt.close()
    print("  [saved] training_curve.png")


def plot_confusion_matrix(cm, model_name, filename):
    _, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax)
    classes = ['Left (0)', 'Right (1)']
    ax.set(xticks=range(2), yticks=range(2),
           xticklabels=classes, yticklabels=classes,
           xlabel='Predicted', ylabel='True',
           title=f'Confusion Matrix — {model_name}')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, filename), dpi=100)
    plt.close()
    print(f"  [saved] {filename}")


def plot_error_examples(images, labels, y_pred, split_idx, n=8):
    """Show images where the scratch model made errors."""
    test_images = images[split_idx:]
    test_labels = labels[split_idx:]
    errors = [(img, true, pred)
              for img, true, pred in zip(test_images, test_labels, y_pred)
              if true != pred]
    if not errors:
        print("  No errors to display!")
        return
    n = min(n, len(errors))
    _, axes = plt.subplots(2, n // 2 + n % 2, figsize=(n * 1.5, 4))
    axes = axes.flatten()
    for k, (img, true, pred) in enumerate(errors[:n]):
        lbl_true = 'Left'  if true == 0 else 'Right'
        lbl_pred = 'Left'  if pred == 0 else 'Right'
        axes[k].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[k].set_title(f'True:{lbl_true}\nPred:{lbl_pred}',
                          fontsize=7, color='red')
        axes[k].axis('off')
    for k in range(n, len(axes)):
        axes[k].axis('off')
    plt.suptitle('Misclassified Examples', fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'error_examples.png'), dpi=100)
    plt.close()
    print("  [saved] error_examples.png")


def plot_feature_importance(model_scratch):
    """Visualize learned weights for the column projections."""
    w = model_scratch.w
    raw_proj_w  = w[:IMG_SIZE]
    edge_proj_w = w[IMG_SIZE:2*IMG_SIZE]

    _, axes = plt.subplots(1, 2, figsize=(10, 4))
    cols = np.arange(IMG_SIZE)
    axes[0].bar(cols, raw_proj_w,
                color=['green' if v > 0 else 'red' for v in raw_proj_w],
                alpha=0.8)
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_title('Weights — Raw Column Projection\n(+green → right, -red → left)')
    axes[0].set_xlabel('Column index')

    axes[1].bar(cols, edge_proj_w,
                color=['green' if v > 0 else 'red' for v in edge_proj_w],
                alpha=0.8)
    axes[1].axhline(0, color='black', linewidth=0.8)
    axes[1].set_title('Weights — Edge Column Projection')
    axes[1].set_xlabel('Column index')

    plt.suptitle('Learned Feature Weights (Scratch Logistic Regression)',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'feature_weights.png'), dpi=100)
    plt.close()
    print("  [saved] feature_weights.png")

def predict_single(img_bgr, model, scaler):
    """
    Predict direction for one BGR image.
    Returns (label_str, prob_right) where prob_right ∈ [0, 1].
    """
    g, b, e = preprocess(img_bgr)
    feat = extract_features(g, b, e).reshape(1, -1)
    feat_s = scaler.transform(feat)
    prob = float(model.predict_proba(feat_s)[0, 1])
    label = 'RIGHT' if prob >= 0.5 else 'LEFT'
    return label, prob


def _draw_overlay(frame, label, prob, roi=None):
    """Render prediction label + confidence bar onto frame in-place (minimalist)."""
    # Optional ROI rectangle
    if roi is not None:
        x, y, w, rh = roi
        cv2.rectangle(frame, (x, y), (x + w, y + rh), (0, 220, 220), 2)

    color = (0, 210, 0) if label == 'RIGHT' else (60, 80, 255)
    arrow = '->' if label == 'RIGHT' else '<-'

    # Dark backing panel
    cv2.rectangle(frame, (8, 8), (220, 80), (20, 20, 20), -1)
    cv2.rectangle(frame, (8, 8), (220, 80), color, 1)

    # Label
    cv2.putText(frame, f'{arrow}  {label}', (16, 48),
                cv2.FONT_HERSHEY_DUPLEX, 1.3, color, 2, cv2.LINE_AA)

    # Probability bar
    bar_x, bar_y, bar_w, bar_h = 16, 58, 190, 12
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + bar_w, bar_y + bar_h), (60, 60, 200), -1)
    fill = int(prob * bar_w)
    cv2.rectangle(frame, (bar_x, bar_y),
                  (bar_x + fill, bar_y + bar_h), (0, 210, 0), -1)
    cv2.putText(frame, f'{prob:.2f}', (bar_x + bar_w + 6, bar_y + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

def run_camera(model, scaler):
    """
    Live webcam inference.
    A yellow rectangle defines the ROI — point your arrow inside it.
    Press 'q' to quit.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[camera] ERROR: cannot open camera (device 0).")
        return

    print("\n[camera] Live inference started.")
    print("  Point an arrow inside the yellow ROI box.")
    print("  Press 'q' to quit.\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        fh, fw = frame.shape[:2]

        # Define a centered square ROI (55 % of frame height)
        roi_size = int(min(fh, fw) * 0.55)
        rx = (fw - roi_size) // 2
        ry = (fh - roi_size) // 2
        roi = (rx, ry, roi_size, roi_size)

        crop = frame[ry:ry + roi_size, rx:rx + roi_size]
        label, prob = predict_single(crop, model, scaler)

        _draw_overlay(frame, label, prob, roi=roi)

        cv2.imshow('Arrow Classifier  [q = quit]', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[camera] Closed.")

def main(args=None):
    images, labels = build_dataset(n_per_class=N_PER_CLASS)
    labels = np.array(labels)

    X = build_feature_matrix(images)

    # ── Train / Test split ───────────────────────────────────
    print("\n[4] Entrenamiento")
    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X, labels, np.arange(len(images)),
        test_size=0.2, random_state=42, stratify=labels
    )
    print(f"  Entrenamiento: {len(X_train)} (80%) | Prueba: {len(X_test)} (20%)")

    # Standardize
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── Logistic Regression 
    print(f"\n[5] Entrenamiento y Evaluación")
    print(f"  Regresión Logística: C=10, solver=lbfgs, max_iter=2000")
    model = SKLearnLR(max_iter=2000, C=10, solver='lbfgs', random_state=42)
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)
    
    acc, prec, rec, f1, cm = evaluate(y_test, y_pred, "Regresión Logística")
    plot_confusion_matrix(cm, "sklearn LR", "cm_sklearn.png")
    
    # Error examples
    test_images_ordered = [images[i] for i in idx_test]
    plot_error_examples(
        test_images_ordered, list(y_test), y_pred,
        split_idx=0
    )

    # Cross-validation
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5, scoring='accuracy')
    print(f"  Validación cruzada (5-fold): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    print(f"\n--- Resumen ---")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  Archivos: error_examples.png, cm_sklearn.png")

    if args is None:
        return

    if args.camera:
        run_camera(model, scaler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Arrow direction classifier (Logistic Regression)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--camera',  action='store_true',
                       help='Live webcam inference after training')
    args = parser.parse_args()
    main(args)
