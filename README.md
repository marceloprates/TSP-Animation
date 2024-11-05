# TSP-Animation

TSP-Animation is a CLI tool (and a Python package) for generating smooth transition animations from unordered collections of images.

It leverages Vision Transformer (ViT) image embeddings and a Travelling Salesman Problem solver to find a sequence of images that (approximately) optimizes the total semantic length, that is, a sequence of images for which the semantic discrepancy between any two adjacent frames is minimized.

# Installation

Install TSP-Animation with
```bash
pip install git+https://github.com/marceloprates/TSP-Animation
```

# Usage

To generate a video (*Videos/my-video.mp4*) from a unordered collection of images contained in the directory *Pictures/my-album/*, use:
```bash
> tspa "Pictures/my-album/" "Videos/my-video.mp4"
```

For an example, use the dataset *Selectarum stirpium Americanarum historia* (from Biodiversity Heritage Library, in the public domain), under the *files/* folder, run:
```bash
> tspa "files/Selectarum stirpium Americanarum historia/" "videos/Selectarum-stirpium-Americanarum-historia.mp4"
```
![TSP Animation Example](videos/Selectarum%20stirpium%20Americanarum%20historia-small.gif)