# april-calib
**Python 3.8+** Uses f-strings and walrus-assignment.
Detect pose of April tags by calibrating camera. 

## Usage

* First run ``calibrate.py`` and press 'c' to capture image of reference tag.
* Default tag is ID-0 of 16H5 family (6.75 cm, 1.00 cm margin).
* Capture atleast 60 images (default; can be changed).
* Specify number of iterations, to estimate intrinsic factors.
* Then run ``tagWrtCam.py``, and press 'd' to get pose of tag printed.
