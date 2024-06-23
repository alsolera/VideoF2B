###########
Field setup
###########

Before you go to the field, please read :doc:`camera-placement` to learn how to place the camera correctly in
the field for best results.

.. important::

    **DO NOT record the videos handheld!**
    
    **ALWAYS** mount the camera to a sturdy tripod or similar.
    
    **DO NOT** move or adjust the camera setup while recording a flight.

For Basic Videos
----------------

To produce :doc:`Basic videos <producing-uncalibrated>` of flights, you only need a suitable camera and a
tripod.

For Augmented-Reality Videos
----------------------------

To produce :doc:`AR videos <producing-calibrated>`, more effort may be required. If your flying site already
has FAI F2B :term:`markers` installed around the :term:`flight circle`, then the surveying work is already
done. Just measure the distance from circle center to the markers.  The elevation of the markers above the
circle center should be 1.5 meters in that case.

    .. image:: images/field-setup-ar.png

Setting Up Your Own Markers
---------------------------

If your field does not have F2B :term:`markers`, you can install them yourself with some basic equipment:

This procedure outlines the steps to place eight markers spaced 45° apart in a 31-meter radius circle, 1.5
meters above the center height, using simple tools.

The recommended dimensions and placement of markers are described in **Annex 4F, Appendix II** of the `FAI
Sporting Code <https://www.fai.org/sites/default/files/sc4_vol_f2_controlline_24.pdf>`__.

.. note::

    31 Meters radius is the FAI recommendation, however, if your circle is smaller than 31 meters, you can
    still place the markers in a smaller circle and indicate it in the "Load a Flight" window. In fact, the
    default value is 25 meters.

Materials Needed
================

- Metric tape measure (at least 45m for 31m markers circle radius)
- Water tube level (about 35m of clear plastic tube)
- Nine sticks (~2 meters in height)
- Chalk or spray paint

Steps
=====

#. **Fix the Center Stick:**

   - Place a vertical stick at the center of the circle.
   - Mark the 1.5-meter height reference on this stick.

#. **Draw the Circle:**

   - Attach one end of the tape measure to the center stick (at ground level).
   - Measure out 31 meters and draw the circle using chalk or spray paint.

    .. figure:: images/markers2.png

        Marking references on circle.

#. **Mark the First Point:**

   - Place a stick at the 0° position on the circumference.

#. **Mark the Opposite Point:**

   - Align by eye, aligning the center stick and the already placed marker to mark the 180° position.
   - Place a stick at this point.

#. **Find the 90° Points:**

   - Use the tape measure to measure :math:`31 \textrm{m} \times \sqrt{2} = 43.84 \textrm{m}` from one of the
     placed points to intersect the circumference.
   - Mark the point located at 90° and place a stick there.
   - Mark the opposite 270° point using the previous method.


    .. figure:: images/markers5.png

        Marking 90 deg references using the diagonal length.

#. **Mark Remaining Points:**

   - You now have four equally spaced points (0°, 90°, 180°, 270°).
   - Before moving the tape from the previous diagonal, measure the half-diagonal, :math:`31 \textrm{m} \times
     \sqrt{2} / 2 = 21.92\textrm{m}` meters, between each adjacent previous points to find the bisectors of
     the 90° angles (i.e., 45°, 135°, 225°, and 315°) and mark the position.
   - Repeat the previous step for the adjacent diagonal.
   - Place the remaining four sticks aligning the center stick and the half diagonal points.

    .. figure:: images/markers6.png

        Marking 45 deg reference points using middle point of diagonals.

#. **Adjust Marker Heights:**

   - Use the water tube level to transfer the 1.5-meter reference height from the center to each of the eight
     marker sticks. Be careful to remove all air bubbles from the tube.

    .. figure:: images/markers7.png

        Tube level schematic.

#. **Place the final markers**

   - Place the final marker plates using the sticks as a reference.

#. **Remove the sticks and enjoy**

.. note::

    This is the simplest and cheapest method we imagined. A laser level or similar equipment will be more
    convenient if it is available.

