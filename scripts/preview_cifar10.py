import os
# временно предотвратим падение, если где-то уже подтянулся другой OMP
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

# важное: неинтерактивный бэкенд без Qt
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import datasets, transforms

classes = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]

out_dir = Path("outputs")
out_dir.mkdir(parents=True, exist_ok=True)
out_path = out_dir / "cifar10_preview.png"

ds = datasets.CIFAR10("data", train=True, download=False, transform=transforms.ToTensor())

fig = plt.figure(figsize=(8,4))
for i in range(12):
    x, y = ds[i]
    ax = fig.add_subplot(3, 4, i+1)
    ax.imshow(x.permute(1,2,0).numpy().clip(0,1))
    ax.set_title(classes[y], fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.savefig(out_path, dpi=150)
print(f"✅ Preview saved to: {out_path}")