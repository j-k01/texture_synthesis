
TEXTURE SYNTHESIS
=================

SETUP:
1. Install requirements: pip install -r requirements.txt
2. Check configurable parameters at top of mainthread.py

USAGE:
1. Place seed image (10_small.png) in same directory as script
2. Run: python mainthread.py

PARAMETERS (edit in script):
- square_size: Output dimensions (default: 400x400)
- WindowSize: Pattern matching window size (default: 32)
- MaxErrThreshold: Pattern matching threshold (default: 0.075)

OUTPUT:
- Final image: synthesized_image_10_32.png
- Progress images in output10_32/