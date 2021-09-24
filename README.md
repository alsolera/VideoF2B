# VideoF2B

Author's blog is [here](http://videof2b.blogspot.com/)

VideoF2B is an open-source desktop application for tracing F2B Control Line Stunt competition flight figures in video.

## Overview

Use this application to trace the path of a control line aircraft as it performs aerobatic maneuvers in F2B competition
and compare it to regulation figures.

Authors: Alberto Solera, Andrey Vasilik

## Features

- Detects the movement of the aircraft and draw the trace of its centroid in video.

- Draws an augmented-reality (AR) hemisphere that represents the flight envelope.

- Displays template figures on the surface of the AR sphere according to
Section 4.2.15 - "Description of Manoeuvres" of the
[FAI Sporting Code (Jan 2021)](https://www.fai.org/sites/default/files/ciam/sc4_vol_f2_controlline_21.pdf).
Manoeuvre diagrams are available in
[Annex 4J (Jan 2021)](https://www.fai.org/sites/default/files/ciam/sc4_vol_f2_controlline_annex_4j_21.pdf)
The latest versions of these regulations are available at the
[FAI Sporting Code page](https://www.fai.org/page/ciam-code) under **Section 4 (Aeromodelling)**.

- Allows the user to rotate and translate the AR sphere during video processing.

- Includes a utility to perform camera calibration. This enables display of the AR sphere in videos.

- Includes a utility to estimate the best camera placement in the field.

## Features (planned)

- Process live video in real time.

- Project the detected points into the virtual sphere in engineering units to track the aircraft in 3D.

- Perform the best possible fit of executed figures to the nominal figures.

- Determine a score per figure.

## Developer installation

### All platforms

- Create a virtual environment.

- Clone the project from this repository and `cd` into the root dir.

- Run `pip install -e .[dev]` in the virtual environment. The extra `[dev]` packages are generally required for all development work, testing, and building releases. They are not required for just running the application from source.

### Linux

- Build OpenCV for the virtual environment based on the instructions [here](https://www.pyimagesearch.com/2018/08/15/how-to-install-opencv-4-on-ubuntu/).

## Building a release

- Tag the latest stable commit in `master` with the desired version using a scheme that complies with PEP 440.

- Switch to the project's root dir.

- Enter the project's virtual environment.

- Run the following commands:

```
    pip install -e .[dev]
    python setup.py build_exe
```

- The first command installs the latest version of the project locally and updates the version file `videof2b/version.py` according to the state of the project's current Git tree.
The second command invokes PyInstaller and builds a binary end-user executable in the `dist` dir.

- Test the executable on target platforms.

- Publish the release to the world.

## External Dependencies

See **setup.cfg**.

**IMPORTANT:** at this time the `imutils` package used for development is a modified fork of the official package.
