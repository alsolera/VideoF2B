# VideoF2B

http://videof2b.blogspot.com/

Application for drawing F2B figures in video.

## Overview

Use this application to trace the path of a control line aircraft as it performs aerobatic maneuvers in F2B competition.

Authors: Alberto Solera, Andrey Vasilik

## Requirements

1. Detect the movement of the aircraft and trace its centroid.

1. Project the detected points into the virtual sphere in engineering units.

1. Perform the best possible fit of executed figures to the nominal figures

1. Determine a score per figure.

## Dev installation

### All platforms

1. Create a virtual environment.

1. Run `pip install -r requirements-dev.txt` in the virtual environment.

### Linux

1. Run `pip install -r requirements.txt` in the virtual environment.

2. Build OpenCV for the virtual environment based on the instructions [here](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/).

### Windows

1. Run `pip install -r requirements_windows.txt` in the virtual environment.

## External Dependencies

* See **requirements.txt**, **requirements-dev.txt**, and **requirements_windows.txt**

* **IMPORTANT:** at this time the `imutils` package used for development is a modified fork of the official package.
To use, execute the following commands from the virtual environment:

  * `pip uninstall imutils`

  * `pip install -U git+https://github.com/basil96/imutils.git`
