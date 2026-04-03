import os
import random
import time
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_curve, auc


# CONFIGURATION 
FONT_DIR = "./generated_pangrams"          # path to extracted Font_Images folder
NUM_FONTS = 300                       # number of fonts to subsample
CROPS_PER_IMAGE = 50                  # random square crops per font image
CROP_SIZE = 32                        # square crop size in pixels
BATCH_SIZE = 256
EMBEDDING_DIM = 128                   # embedding dimension for prototypical eval
NUM_EPOCHS = 60
LEARNING_RATE = 1e-3
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# DEVICE SETUP (MPS for Mac; otherwise fallback to CPU)
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple MPS (Metal) acceleration")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")


# IMAGE PREPROCESSING AND DATA CONSTRUCTION
print("\n" + "="*60)
print("STEP 1: Image Preprocessing and Data Construction")
print("="*60)

# Crop functions (split a 400x400 font image into 2 sentence crops)
def crop_sentences(img):
    raw_pix = np.array(img)
    preprocess_im = []
    for pg_line in [0, 1]:
        pix = raw_pix[pg_line * 200:200 + pg_line * 200, :]
        topcol = np.argmax((np.argmax(pix != 255, axis=0) > 0))
        botcol = np.argmax((np.argmax(np.flip(pix, axis=(0, 1)) != 255, axis=0) > 0))
        toprow = np.argmax((np.argmax(pix != 255, axis=1) > 0))
        botrow = np.argmax((np.argmax(np.flip(pix, axis=(0, 1)) != 255, axis=1) > 0))
        pix = pix[toprow:200 - botrow, topcol:400 - botcol]
        crop_img = Image.fromarray(pix)
        preprocess_im.append(crop_img)
    return preprocess_im

# Generate random square crops from a sentence image 
def random_square_crops(sentence_img, crop_size=32, n_crops=5):
    arr = np.array(sentence_img)
    h, w = arr.shape
    # Pad if smaller than crop_size
    if h < crop_size or w < crop_size:
        padded = np.full((max(h, crop_size), max(w, crop_size)), 255, dtype=np.uint8)
        padded[:h, :w] = arr
        arr = padded
        h, w = arr.shape
    crops = []
    for _ in range(n_crops):
        y = np.random.randint(0, h - crop_size + 1)
        x = np.random.randint(0, w - crop_size + 1)
        crop = arr[y:y + crop_size, x:x + crop_size]
        crops.append(crop)
    return crops


#  Load and subsample fonts 
all_files = sorted(glob.glob(os.path.join(FONT_DIR, "*.bmp")))
print(f"Total font images found: {len(all_files)}")

if len(all_files) == 0:
    raise FileNotFoundError(
        f"No .bmp files found in {FONT_DIR}. "
        f"Please extract Font_Images.zip and set FONT_DIR correctly."
    )

# Subsample fonts
sampled_files = random.sample(all_files, min(NUM_FONTS, len(all_files)))
print(f"Subsampled to {len(sampled_files)} fonts")

# Generate crops and labels 
print("Generating crops...")
all_crops = []  
font_names = []  # store font IDs for reference
skipped = 0

for label_idx, filepath in enumerate(sampled_files):
    font_id = os.path.basename(filepath).replace(".bmp", "")
    font_names.append(font_id)

    try:
        img = Image.open(filepath)
        sentences = crop_sentences(img)

        crops_per_sentence = CROPS_PER_IMAGE // 2
        for sentence in sentences:
            sq_crops = random_square_crops(
                sentence, crop_size=CROP_SIZE, n_crops=crops_per_sentence
            )
            for crop in sq_crops:
                all_crops.append((crop, label_idx))
    except Exception as e:
        skipped += 1
        if skipped <= 5:
            print(f"  Warning: skipped {font_id}: {e}")

print(f"Total crops generated: {len(all_crops)}")
print(f"Fonts skipped: {skipped}")
print(f"Crops per font: {CROPS_PER_IMAGE}")
print(f"Number of classes: {len(font_names)}")

# Shuffle and split into train/val/test
random.shuffle(all_crops)

X_all = np.array([c[0] for c in all_crops], dtype=np.float32) / 255.0  # normalise to [0,1]
y_all = np.array([c[1] for c in all_crops], dtype=np.int64)

n_total = len(X_all)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

X_train, y_train = X_all[:n_train], y_all[:n_train]
X_val, y_val = X_all[n_train:n_train + n_val], y_all[n_train:n_train + n_val]
X_test, y_test = X_all[n_train + n_val:], y_all[n_train + n_val:]

print(f"\nSplit sizes:")
print(f"  Train: {len(X_train)}")
print(f"  Val:   {len(X_val)}")
print(f"  Test:  {len(X_test)}")



# PyTorch Dataset
class FontCropDataset(Dataset):
    def __init__(self, X, y):
        # X: (N, H, W) float32 in [0,1], y: (N,) int64
        self.X = torch.tensor(X).unsqueeze(1)  # (N, 1, H, W)
        self.y = torch.tensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = FontCropDataset(X_train, y_train)
val_dataset = FontCropDataset(X_val, y_val)
test_dataset = FontCropDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)



# MODEL DEFINITION AND TRAINING
print("\n" + "="*60)
print("STEP 2: Model Definition and Training")
print("="*60)


class FontCNN(nn.Module):
    """
    CNN for font classification with an embedding layer.

    Architecture:
      Conv2d(1, 32, 3, padding=1) → BatchNorm → ReLU → MaxPool(2)
      Conv2d(32, 64, 3, padding=1) → BatchNorm → ReLU → MaxPool(2)
      Conv2d(64, 128, 3, padding=1) → BatchNorm → ReLU → MaxPool(2)
      Flatten → Linear(128*4*4, 256) → ReLU → Dropout(0.3)
      Linear(256, EMBEDDING_DIM) → ReLU        [embedding layer]
      Linear(EMBEDDING_DIM, num_classes)        [classifier head]

    The embedding is extracted from the second-to-last layer
    for prototypical evaluation.
    """

    def __init__(self, num_classes, embedding_dim=128):
        super(FontCNN, self).__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 → 16x16
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 16x16 → 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 8x8 → 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)

        self.embedding = nn.Linear(256, embedding_dim)
        self.relu2 = nn.ReLU(inplace=True)

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(self.relu1(self.fc1(x)))
        emb = self.relu2(self.embedding(x))
        out = self.classifier(emb)
        return out

    def get_embedding(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu1(self.fc1(x)))
        emb = self.relu2(self.embedding(x))
        return emb


num_classes = len(font_names)
model = FontCNN(num_classes=num_classes, embedding_dim=EMBEDDING_DIM).to(device)

print(f"\nModel architecture:")
print(model)
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# Training setup 
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=3
)

# Training loop 
train_losses = []
val_losses = []
train_accs = []
val_accs = []
best_val_acc = 0
best_model_state = None

print(f"\nTraining for {NUM_EPOCHS} epochs...")
print(f"Batch size: {BATCH_SIZE}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"Optimizer: Adam (weight_decay=1e-4)")
print(f"Scheduler: ReduceLROnPlateau (patience=3, factor=0.5)")
print()

start_time = time.time()

for epoch in range(NUM_EPOCHS):
    # Train 
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * batch_X.size(0)
        _, predicted = outputs.max(1)
        total += batch_y.size(0)
        correct += predicted.eq(batch_y).sum().item()

    train_loss = running_loss / total
    train_acc = correct / total
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # Validate 
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            running_loss += loss.item() * batch_X.size(0)
            _, predicted = outputs.max(1)
            total += batch_y.size(0)
            correct += predicted.eq(batch_y).sum().item()

    val_loss = running_loss / total
    val_acc = correct / total
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    scheduler.step(val_loss)

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_state = model.state_dict().copy()

    if (epoch + 1) % 5 == 0 or epoch == 0:
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{NUM_EPOCHS} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.0f}s")

total_time = time.time() - start_time
print(f"\nTraining complete in {total_time:.1f}s ({total_time/60:.1f} min)")
print(f"Best validation accuracy: {best_val_acc:.4f}")

# Load best model
model.load_state_dict(best_model_state)

# Plot training curves 
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(train_losses, label="Train")
ax1.plot(val_losses, label="Validation")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("Training and Validation Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(train_accs, label="Train")
ax2.plot(val_accs, label="Validation")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy")
ax2.set_title("Training and Validation Accuracy")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: training_curves.png")



# EVALUATION
print("\n" + "="*60)
print("STEP 3: Evaluation")
print("="*60)

# Extract all test embeddings 
print("\nExtracting test set embeddings...")
model.eval()
all_embeddings = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        emb = model.get_embedding(batch_X)
        all_embeddings.append(emb.cpu())
        all_labels.append(batch_y)

all_embeddings = torch.cat(all_embeddings, dim=0)  # (N_test, EMBEDDING_DIM)
all_labels = torch.cat(all_labels, dim=0)           # (N_test,)
print(f"Test embeddings shape: {all_embeddings.shape}")

# Top-1 and Top-5 Accuracy (Prototypical Network Method) 
print("\nComputing top-1 and top-5 accuracy (prototypical network method)...")

# Group embeddings by class
class_embeddings = defaultdict(list)
for i in range(len(all_labels)):
    c = all_labels[i].item()
    class_embeddings[c].append(all_embeddings[i])

# Only evaluate classes with at least 2 test samples (need query + support)
valid_classes = [c for c, embs in class_embeddings.items() if len(embs) >= 2]
print(f"Classes with ≥2 test samples: {len(valid_classes)} / {num_classes}")

top1_correct = 0
top5_correct = 0
total_queries = 0

for c in valid_classes:
    embs = class_embeddings[c]

    # Randomly select one as query, rest as support
    query_idx = random.randint(0, len(embs) - 1)
    query = embs[query_idx]
    support = torch.stack([embs[j] for j in range(len(embs)) if j != query_idx])

    # Prototype = mean of support embeddings
    prototype_c = support.mean(dim=0)

    # Compute cosine similarity between query and all class prototypes
    similarities = []
    for other_c in valid_classes:
        other_embs = class_embeddings[other_c]
        if other_c == c:
            proto = prototype_c
        else:
            proto = torch.stack(other_embs).mean(dim=0)

        # Cosine similarity
        cos_sim = torch.nn.functional.cosine_similarity(
            query.unsqueeze(0), proto.unsqueeze(0)
        ).item()
        similarities.append((other_c, cos_sim))

    # Sort by similarity (highest first)
    similarities.sort(key=lambda x: x[1], reverse=True)

    # Top 1
    if similarities[0][0] == c:
        top1_correct += 1

    # Top 5
    top5_classes = [s[0] for s in similarities[:5]]
    if c in top5_classes:
        top5_correct += 1

    total_queries += 1

top1_acc = top1_correct / total_queries
top5_acc = top5_correct / total_queries

print(f"\nPrototypical Network Evaluation Results:")
print(f"  Top-1 Accuracy: {top1_acc:.4f} ({top1_correct}/{total_queries})")
print(f"  Top-5 Accuracy: {top5_acc:.4f} ({top5_correct}/{total_queries})")


# Pairwise ROC Curve 
print("\nComputing pairwise ROC curve...")
print("(This may take a moment for large test sets...)")

# Sample pairs if test set is large
N_test = len(all_embeddings)
MAX_PAIRS = 100000  # cap to keep memory reasonable

# Generate positive and negative pair similarities
pos_sims = []  # same class
neg_sims = []  # different class

if N_test * (N_test - 1) // 2 <= MAX_PAIRS:
    # Compute all pairs
    for i in range(N_test):
        for j in range(i + 1, N_test):
            sim = torch.nn.functional.cosine_similarity(
                all_embeddings[i].unsqueeze(0),
                all_embeddings[j].unsqueeze(0)
            ).item()
            if all_labels[i] == all_labels[j]:
                pos_sims.append(sim)
            else:
                neg_sims.append(sim)
else:
    # Sample pairs
    print(f"  Sampling {MAX_PAIRS} pairs (full set too large)...")
    indices = list(range(N_test))
    sampled = 0
    while sampled < MAX_PAIRS:
        i, j = random.sample(indices, 2)
        sim = torch.nn.functional.cosine_similarity(
            all_embeddings[i].unsqueeze(0),
            all_embeddings[j].unsqueeze(0)
        ).item()
        if all_labels[i] == all_labels[j]:
            pos_sims.append(sim)
        else:
            neg_sims.append(sim)
        sampled += 1

print(f"  Positive pairs (same font): {len(pos_sims)}")
print(f"  Negative pairs (diff font): {len(neg_sims)}")

# Compute ROC by sweeping threshold
all_sims = pos_sims + neg_sims
all_pair_labels = [1] * len(pos_sims) + [0] * len(neg_sims)

all_sims = np.array(all_sims)
all_pair_labels = np.array(all_pair_labels)

fpr, tpr, thresholds = roc_curve(all_pair_labels, all_sims)
roc_auc = auc(fpr, tpr)

print(f"  AUC: {roc_auc:.4f}")

# Plot ROC curve 
plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, color='#378ADD', linewidth=2,
         label=f'Pairwise ROC (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], '--', color='gray', linewidth=1)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve — Pairwise Cosine Similarity')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve_pairwise.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: roc_curve_pairwise.png")



# SUMMARY
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Dataset: {NUM_FONTS} fonts subsampled from {len(all_files)} total")
print(f"Crops per font: {CROPS_PER_IMAGE} ({CROPS_PER_IMAGE//2} per sentence line)")
print(f"Crop size: {CROP_SIZE}x{CROP_SIZE} pixels, grayscale")
print(f"Total samples: {n_total} (train: {n_train}, val: {n_val}, test: {n_total - n_train - n_val})")
print(f"")
print(f"Model: FontCNN (3-block CNN → 256 → {EMBEDDING_DIM} embedding → {num_classes} classes)")
print(f"Parameters: {total_params:,}")
print(f"Optimizer: Adam (lr={LEARNING_RATE}, weight_decay=1e-4)")
print(f"Scheduler: ReduceLROnPlateau")
print(f"Training: {NUM_EPOCHS} epochs, batch size {BATCH_SIZE}")
print(f"Training time: {total_time:.1f}s")
print(f"Best validation accuracy: {best_val_acc:.4f}")
print(f"")
print(f"Top-1 accuracy (prototypical): {top1_acc:.4f}")
print(f"Top-5 accuracy (prototypical): {top5_acc:.4f}")
print(f"Pairwise ROC AUC: {roc_auc:.4f}")