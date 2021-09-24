# VideoF2B

[Project Blog](http://videof2b.blogspot.com/)

VideoF2B is an open-source desktop application for tracing F2B Control Line Stunt competition flight figures in video.

## Overview

Use this application to trace the path of a control line aircraft as it performs aerobatic maneuvers in F2B competition
and compare it to regulation figures.

Authors: Alberto Solera, Andrey Vasilik

## Features

1. Detects the movement of the aircraft and draw the trace of its centroid in video.

1. Draws an augmented-reality (AR) hemisphere that represents the flight envelope.

1. Displays template figures on the surface of the AR sphere according to
Section 4.2.15 - "Description of Manoeuvres" of the
[FAI Sporting Code (Jan 2021)](https://www.fai.org/sites/default/files/ciam/sc4_vol_f2_controlline_21.pdf).
Manoeuvre diagrams are available in
[Annex 4J (Jan 2021)](https://www.fai.org/sites/default/files/ciam/sc4_vol_f2_controlline_annex_4j_21.pdf)
The latest versions of these regulations are available at the
[FAI Sporting Code page](https://www.fai.org/page/ciam-code) under **Section 4 (Aeromodelling)**.

1. Allows the user to rotate and translate the AR sphere during video processing.

1. Includes a utility to perform camera calibration. This enables display of the AR sphere in videos.

1. Includes a utility to estimate the best camera placement in the field.

## Features (planned)

1. Process live video in real time.

1. Project the detected points into the virtual sphere in engineering units to track the aircraft in 3D.

1. Perform the best possible fit of executed figures to the nominal figures.

1. Determine a score per figure.

## Developer installation

### All platforms

1. Create a virtual environment.

1. Clone the project from this repository and `cd` into the root dir.

1. Run `pip install -e .` in the virtual environment.

### Linux

1. Build OpenCV for the virtual environment based on the instructions [here](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/).

## External Dependencies

* See **setup.cfg**.

* **IMPORTANT:** at this time the `imutils` package used for development is a modified fork of the official package.
