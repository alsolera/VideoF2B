.. FAQ for users of VideoF2B

###
FAQ
###

**Q: Why do I need markers?**

**A:** F2B markers provide a base reference in the field for the pilot and for the judges.  For VideoF2B, they
also relate the real world to camera images, so that augmented reality geometry can be drawn accurately in
video.

-----

**Q: My flying site does not have F2B markers, and installing them is not practical.  Is there an alternative
method for creating AR videos that does not require markers?**

**A:** This capability is a research & development project that is currently in progress.

-----

**Q: Can I move the camera during a flight once recording has started?**

**A:** No. Doing so will require you to select the markers in video in the new camera location.

-----

**Q: Does the trace drawn in video follow the CG of the model aircraft?**

**A:** Not necessarily, but it tends to be fairly close to it.  The motion detector follows the `centroid
<https://en.wikipedia.org/wiki/Centroid>`__ (geometric center) of the silhouette of the largest moving object
in video.

-----

**Q: Why are parts of the motion trace sometimes missing?**

**A:** The motion detector follows the silhouette of the largest moving object in the field of view.  When the
only moving object in the FOV is the model aircraft, the trace remains continuous. However, when objects
larger than the model aircraft move within the FOV (e.g., vehicles in the background, people, field insects,
etc.), those objects "steal" the trace.  This is the current behavior of the motion detector in VideoF2B.  It
may be improved in the future.

-----

**Q: How accurate are the Augmented Reality graphics?**

**A:** Field tests have proven that the augmented-reality geometry drawn in video is accurate within `10 cm`
throughout the entire flight envelope.

-----

**Q: Can this system be used for computerized scoring of Stunt flights?**

**A:** The concept of tracking the flights in three dimensions using recorded video is definitely under
consideration.  Some maneuvers are problematic to track accurately (takeoff, level/inverted flight, landing),
but the majority of the flight maneuvers are potential candidates.

