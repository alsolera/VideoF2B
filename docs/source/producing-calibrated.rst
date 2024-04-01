##################################
Producing Augmented-Reality Videos
##################################

An :term:`Augmented-Reality <Augmented Reality>` (AR) video contains overlays of various reference graphics on
top of the original video footage.  In addition to the motion trace behind the model aircraft, the AR graphics
may include the following:

- A wireframe representation of the :term:`flight hemisphere`, which includes :term:`verticals <vertical>` at
  every 1/8 lap, the 45° :term:`horizontal`, tolerance horizontals above and below the base.
- Marker points.
- :term:`Nominal` figure representations according to current rules.
- Start and end points of maneuvers.
- Diagnostic points in figures.

Here is an example:

    .. image:: images/calibrated-example-snap.png

To produce an AR video, follow these steps:

#. :doc:`Calibrate your camera <camera-calibration>`.

#. To enable AR graphics, see :doc:`field-setup`.

#. Record a Stunt flight using a video camera. For guidelines on how to position the camera in the field, see
   :doc:`camera-placement`. Save the video file to your computer.

#. :doc:`Load the flight <loading-flight>` into VideoF2B.

#. :doc:`Locate the flight <locating-flight>` in VideoF2B.

#. Process the flight video. See :doc:`user-controls-ar`.

#. When finished, you will find the processed AR video file in the same location as the original video. The AR
   video will have the same name as the original, but with a ``_out`` suffix.  For example, if your original
   video is named ``Flight 1.mp4``, the traced video will be named ``Flight 1_out.mp4``.
