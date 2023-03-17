Go Where?
=

This repository is a very WIP prototype for detection of go boards in images.
Eventually, this will allow you to to quickly detect a go board from an image and interpret its state.

## Method
 1. (WIP) Detect corners of the Go board
 2. (TODO) Compute (non-unique) camera transformation
 3. (TODO) Transform detected Go board grid to "standard" grid
 4. (TODO) Read game state

### Corner Detection
We use a simplyfied version of the [YOLO](https://arxiv.org/abs/1506.02640) algorithm.
This setting is simpler in two major ways: Corners are single points, i.e. we don't require bounding boxes, only locations and further we only have one class that is either present or not.
What makes this setting slightly more challenging is that we need the corner point location predictions to be very precise.

