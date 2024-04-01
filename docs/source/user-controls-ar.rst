#############
User Controls
#############

This chapter applies to the user controls that are available during
:doc:`Augmented-Reality (AR) processing <producing-calibrated>`.

The following user controls on the left side of the main window are enabled during AR processing:

    .. figure:: images/user-controls-ar.png

        User controls available during AR processing.

VideoF2B reads frames from a given video in strict sequence from beginning to end. The user interface is
designed for an efficient workflow via the keyboard alone, while some of those controls are also available in
the main window, as seen above.

Pausing/Resuming Processing
---------------------------

There is no fast-forward or rewind functionality. However, you can pause the processing at any time by
pressing :kbd:`P` on the keyboard or clicking the |Icon for Pause button| button.  While processing is paused,
you can perform various manipulations of AR geometry, taking as much time as you need. When ready to continue,
press :kbd:`P` again or click the |Icon for Play button| button.

Clearing the Trace
------------------

To clear the trace behind the model aircraft, press :kbd:`Space`. This is useful for presenting clear traces
of maneuvers — clear the trace shortly before an upcoming maneuver to present the traced maneuver clearly.

The World Coordinate System
---------------------------

**TODO**: move this heading to its own chapter after :doc:`locating-flight`?

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

Manipulating the Flight Hemisphere
----------------------------------

To account for the pilot's movement in the pilot circle during a flight, use the `WASD keys
<https://www.computerhope.com/jargon/w/wsad.htm>`__ to move the flight hemisphere in the world XY plane. The
keys operate as follows:

    :kbd:`W` moves the hemisphere in **+Y** direction (**forward**, away from the camera).

    :kbd:`S` moves the hemisphere in **-Y** direction (**backward**, toward the camera).

    :kbd:`A` moves the hemisphere in **-X** direction (to the **left**).

    :kbd:`D` moves the hemisphere in **+X** direction (to the **right**).

Every stroke of the above keys moves the hemisphere in the commanded direction by **0.1 m**.

Pressing :kbd:`X` resets the hemisphere's center to the origin of the World Coordinate System.

When the hemisphere's center is not at the origin, its offset will be displayed in the bottom left corner of
the video window:

    .. figure:: images/sphere-info-overlay.png

        Sphere information overlay in video. All dimensions are in meters.

.. tip:: *Dealing with the pilot's off-center displacement*

    Position of the pilot along the X-axis is easy to match accurately. Position along the Y-axis is more
    difficult to estimate because depth is difficult to gauge in video. Take advantage of the **Reverse
    Wingover** maneuver to assess the pilot's initial position because you will be able to adjust the
    hemisphere's position so that the aircraft's centerline crosses the visible edge of the sphere. As the
    flight proceeds, use your best judgment. Other maneuvers whose approaches cross the visible edge of the
    hemisphere above the base (**Outside Square Loops** and **Overhead Eight**) also help to correct for the
    pilot's position along the Y-axis throughout the flight.

To match the :term:`nominal` figure to the maneuver flown by the pilot, use the arrow keys to rotate the
hemisphere. The keys operate as follows:

    :kbd:`Left Arrow` rotates the AR hemisphere **counterclockwise** on its vertical axis (i.e., the nominal
    figure moves to the **left** as seen by the pilot).

    :kbd:`Right Arrow` rotates the AR hemisphere **clockwise** on its vertical axis (i.e., the nominal figure
    moves to the **right** as seen by the pilot).

Every stroke of these arrow keys rotates the hemisphere in the commanded direction by **0.5°**.

Displaying Nominal Figures
------------------------------

**TODO**

Displaying Start/End Points
---------------------------

**TODO**

Displaying Diagnostic Points
----------------------------

**TODO**



.. |Icon for Pause button| image:: images/icons/pause-circle-line.svg
    :class: inline

.. |Icon for Play button| image:: images/icons/play-circle-line.svg
    :class: inline
