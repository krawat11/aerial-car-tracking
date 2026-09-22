# Project site (GitHub Pages)

Static showcase page. To publish: repository Settings -> Pages -> Source: "Deploy from a branch",
branch `main`, folder `/docs`. The site appears at https://krawat11.github.io/aerial-car-tracking/

## Media to add

- `media/demo.mp4` - a short (20-40 s) compressed clip from `runs/videos/`. Keep it under ~25 MB:
  `ffmpeg -i runs/videos/CLIP.mp4 -vf scale=1280:-2 -c:v libx264 -crf 28 -preset slow -an docs/media/demo.mp4`
- `media/poster.jpg` - a still frame for before the video plays:
  `ffmpeg -i docs/media/demo.mp4 -ss 3 -frames:v 1 docs/media/poster.jpg`
- `media/training.png` - the training curves (already copied from models/).
