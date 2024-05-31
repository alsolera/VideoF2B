###########################
The World Coordinate System
###########################

The :doc:`flight locating <locating-flight>` procedure establishes a World Coordinate System (WCS) based on
the selected markers. The WCS is a `right-handed <https://en.wikipedia.org/wiki/Right-hand_rule>`__ `Cartesian
<https://en.wikipedia.org/wiki/Cartesian_coordinate_system>`__ coordinate system. VideoF2B uses the WCS to
draw all AR graphics correctly in video. The position and orientation of the WCS is as follows:

- Origin is at the center of the :term:`base`. Thus, its elevation is 1.5 m above the pilot circle.
- Positive Y-axis passes through the **front marker**.
- Positive Z-axis points vertically upward, and passes through :term:`top of circle`.
- Positive X-axis is perpendicular to both Y and Z axes, and generally points to the right in video.

    **TODO**: replace the photo below with a field photo that shows:
      - all four markers and the camera on a tripod outside the circle.
      - SVG graphics that depict the WCS.
      - AR graphics that depict the flight hemisphere.

    .. image:: images/todo/markers-arrangement.jpg

