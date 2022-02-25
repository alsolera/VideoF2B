.. VideoF2B documentation of the math behind the geometry of nominal figures.

.. meta::
   :keywords: videof2b, geometry, control line, figures, maneuvers
   :description lang=en: VideoF2B documentation of the math behind the geometry of nominal figures.

###############
Figure Geometry
###############

Introduction
============

Geometry is an exact discipline. When we describe pure geometrical elements there is no allowance for error,
and hence no concept of dimensional tolerance. A point is infinitesimally small; a line is infinitely long
and infinitesimally thin; a plane is perfectly flat, infinitesimally thin, and its surface extends in all
directions to infinity. Manipulations of such geometrical elements yield other elements that retain this
general precision. For example, the intersection of two nonparallel planes yields a line of infinitesimal
thickness and infinite length.

On the other hand, the physical world around us is imperfect. Rigid bodies are not truly rigid; they change
shape and size with changes in temperature and pressure. Motion in a perfectly straight line or circle is not
actually possible. Regardless, we find the perfection of geometry and the purity of mathematics to be
suitable tools for representation and measurement of imprecise, real physical objects. In practice, we
typically reconcile the conflict between these two realms by allowing the size and form of geometrical
entities to deviate from their prescribed ideal definition.

All flight paths in Control Line Stunt maneuvers take place on a sphere. In terms of ideal geometry the
center of the sphere is fixed in space and has a fixed radius. We define stunt maneuvers in precise terms as
if we were drawing infinitesimally thin curves on the surface of the perfect sphere. In reality the center of
the sphere moves as the pilot walks within the pilot's circle, and the sphere's radius changes as the control
lines sag more or less under influence of line tension, gravity, and air resistance. Therefore, a precise
representation of ideal stunt :term:`figures <figure>` using **nominal** paths is necessary so that we can assess the amount
of deviation of an actual executed :term:`maneuver` from its nominal.

Prerequisites
=============

Before diving into this guide, make sure you have a fair knowledge of the following topics:

- Algebra_
- Trigonometry_
- `Euclidean geometry <euclidean-geometry_>`_
- `Linear algebra <linear-algebra_>`_
- some Python_ programming :)

This document touches all of these areas. Note that for brevity, not all proofs are explained here. For a more
rigorous treatment of the various topics, see the related resources.

Coordinate Systems
==================

A clear definition of a coordinate system is required for any intelligent discussion about precise geometric
shapes in 3D space. Two systems are relevant here: **Cartesian** and **spherical**.

.. rubric:: Cartesian

.. _fig-cs-cartesian:

.. rst-class:: centered
.. tikz:: Cartesian coordinate system
    :include: tex/cs-cartesian.tex

This is a standard set of three mutually perpendicular axes arranged according to the right-hand rule.
The origin is located in the center of the flight circle at the height of the flight base.
Orientation is relative to the video camera. The positive :math:`x` axis points to the camera's right,
positive :math:`y` axis away from camera, and positive :math:`z` axis up. We describe a point
:math:`\mathbf{\vec{P}}(x, y, z)` as a vector by projecting it orthogonally onto each of the coordinate axes.

.. rubric:: Spherical

.. _fig-cs-spherical:

.. rst-class:: centered
.. tikz:: Spherical coordinate system using elevation
    :include: tex/cs-spherical-elevation.tex

This is a standard spherical coordinate system as commonly used in physics with one modification for
convenience: **elevation** instead of declination. Note that this system builds upon the Cartesian coordinate
system. We describe a point :math:`\mathbf{\vec{P}}(r, \theta, \phi)` as a vector in terms of three
quantities:

- The straight-line distance :math:`r` from :math:`\mathbf{\vec{P}}` to the origin;
- The elevation angle :math:`\theta` formed between the :math:`xy` plane and :math:`\mathbf{\vec{P}}`,
  positive when above the :math:`xy` plane.
- The azimuth angle :math:`\phi` formed between the positive :math:`x` axis and the projection of
  :math:`\mathbf{\vec{P}}` onto the :math:`xy` plane, positive counterclockwise as viewed from the top;

Coordinate Conversions
======================

It is often necessary to convert between the two coordinate systems.

.. rubric:: Spherical to Cartesian

Given :math:`\mathbf{\vec{P}}(r, \theta, \phi)` in spherical coordinates, we express
:math:`\mathbf{\vec{P}}(x, y, z)` in Cartesian coordinates as follows:

.. math::
    \begin{align}
    x& = r \cos\theta \cos\phi\\
    y& = r \cos\theta \sin\phi\\
    z& = r \sin\theta
    \end{align}

.. rubric:: Cartesian to Spherical

Given :math:`\mathbf{\vec{P}}(x, y, z)` in Cartesian coordinates, we express
:math:`\mathbf{\vec{P}}(r, \theta, \phi)` in spherical coordinates as follows:

.. math::
    \begin{align}
    r& = \sqrt{x^2 + y^2 + z^2}\\
    \theta& = \arctan{ \frac{z}{\sqrt{x^2 + y^2}} }\\
    \phi& = \arctan{ \left( \frac{y}{x} \right) }
    \end{align}

.. admonition:: The :math:`\arctan` functions in Python

    Here, the ``math.atan2`` [#atan2]_ function is best suited for :math:`\arctan` in lieu of
    ``math.atan`` [#atan]_ because ``atan2`` considers the signs of its arguments, always placing the
    resulting angle in the correct quadrant.

Spherical Geometry
==================

Some basic understanding of spherical geometry is required. One excellent resource is
*Spherical Trigonometry: For the Use of Colleges and Schools* (Todhunter, Isaac, 1886), freely available from
`Project Gutenberg <ebook-todhunter_>`_. Let's discuss a few concepts in spherical geometry that are relevant
to Control Line flight.

#. A :term:`sphere` is a solid bounded by a surface every point of which is equally distant from a fixed point
   which is called the **center** of the sphere. The straight line that joins any point of the surface with
   the center is called a **radius**. A straight line drawn through the center and terminated both ways by the
   surface is called a **diameter**.

#. *The section of the surface of a sphere made by any plane is a circle.*

   .. _fig-circle-of-sphere:

   .. figure:: images/011fc.png
        :height: 200px
        :align: center

        Circle of a sphere

   Let :math:`AB` be the section of the surface of a sphere made by any plane, :math:`O` the center of the
   sphere. Draw :math:`OC` perpendicular to the plane; take any point :math:`D` in the section and join
   :math:`OD`, :math:`CD`. Since :math:`OC` is perpendicular to the plane, the angle :math:`OCD` is a right
   angle; therefore :math:`CD=\sqrt{OD^2-OC^2}`. Now :math:`O` and :math:`C` are fixed points, so that
   :math:`OC` is constant; and :math:`OD` is constant, being the radius of the sphere; hence :math:`CD` is
   constant. Thus, all points in the plane section are equally distant from the fixed point :math:`C`;
   therefore the section is a circle of which :math:`C` is the center.

#. The section of the surface of a sphere by a plane is called a :term:`great circle <Great circle>` if the
   plane passes through the center of the sphere, and a :term:`small circle <Small circle>` if the plane does
   not pass through the center of the sphere. Thus, the radius of a great circle is equal to the radius of the
   sphere.

#. The angle between two great circles is the angle between the planes of the circles.

#. *A small circle is the base of a right circular cone whose apex is at the center of the sphere.*
   In :numref:`fig-circle-of-sphere`, let :math:`Q` be the surface formed by sweeping :math:`OD` around
   :math:`OC`. The surface :math:`Q` is the :term:`ruled surface` of a cone whose **apex** is the center of
   the sphere. Hence, :math:`OC` is the **axis** of the cone. The small circle is the **base** of the cone.
   The radius of the small circle is the base radius of the cone, and is often simply called the radius of the
   cone. The perimeter of the small circle is called the **directrix**. Each of the line segments between the
   directrix and the apex is a **generatrix**, or a "generating line" of the cone's ruled surface. The maximum
   angle between two generatrix lines is the **aperture** or **apex angle** of the cone. If the generatrix
   makes an angle :math:`\beta` with the axis of the cone, then the aperture is :math:`2\beta`. The cone in
   this instance is called a :term:`right circular cone`.

#. **TODO**: spherical triangles, their terminology, etc.

Transformations
===============

All F2B maneuvers and figures are inherently three-dimensional entities. It is possible to parameterize their
constituent pieces directly. However, it is much easier to begin with simple arcs and circles in 2D and to
arrive at the desired 3D entity via a series of rotations and translations. These rotations and translations
are a subset of the more general **linear transformations** in Euclidean space.

**TODO**: describe utilization of the function ``scipy.spatial.transform.Rotation.from_euler``. Include example with pictures.

Figures
=======

A competition F2B pattern consists of fifteen (15) maneuvers. Of these, the following cannot be easily tracked
in video due to certain limitations:

.. hlist::
    :columns: 2

    - Take-off
    - Reverse Wing-over
    - Inverted flight
    - Landing 

On the other hand, the geometry of the remaining eleven (11) maneuvers can be precisely drawn in video. Let's
study them in detail.

.. admonition:: If you remember nothing else...

    **Remember this:**

    The most basic element of **every** F2B figure is the **arc**. When we break down any maneuver into its
    most elementary pieces, we discover this important truth: all the pieces are either **great arcs** or
    **small arcs**. A great arc is part of a great circle, and a small arc is part of a small circle.
    In this context, a circle is just a special arc whose included angle is :math:`2\pi`.

Loop
----

The basic round loop is a :term:`small circle <Small circle>`. Its path is the directrix of a cone whose apex
is the center of the sphere. If the control lines of the aircraft could be straight lines, they would form the
generatrix, i.e., they would sweep the ruled surface of the cone. We define the angle of the loop to be the
apex angle of the cone, and the elevation of the loop to be the elevation of the axis of the cone. The angle
of all round loops except of those that are part of the four-leaf clover, is 45 degrees.

Corners of figures such as square loops, triangles, and the hourglass are arcs of the basic round loop.

The basic loop is so prevalent in F2B maneuvers that the helper function
:py:func:`get_arc <videof2b.core.geometry.get_arc>` is dedicated to its creation in VideoF2B. This function
creates a discrete set of points that represent a counterclockwise arc of specified radius and included angle
in the :math:`xy` plane. The first point of the arc always lies on the :math:`x` axis.

**TODO**: more content...

Fillet
------

What is a fillet? Why do we need it?

**TODO**: content...

Square
------

**TODO**: content...

Triangle
--------

**TODO**: content...

Hourglass
---------

**TODO**: content...

Four-leaf Clover
----------------

**TODO**: content...

Glossary
========

.. glossary::

    Figure
        A shape that makes up a separately recognizable complete part of a whole :term:`maneuver`. For
        example, the first loop of the three consecutive inside loops maneuver is referred to as a
        :term:`figure`; but the first loop that makes the first half of the first complete figure eight in the
        "two consecutive overhead eight" maneuver is not referred to as a figure.

    Great circle
        The nontrivial intersection of a plane and a :term:`sphere` such that the plane contains the center of
        the sphere.

    Maneuver
        The full total of :term:`figures <figure>` and :term:`segments <segment>` necessary to complete the
        maneuver marked under a separate numbered heading with bold type in the Rules. For example, the
        take-off maneuver, the "three consecutive inside loops" maneuver, and the single "four-leaf clover"
        maneuver, are all referred to as a single whole maneuver.

    Right circular cone
        The commonly assumed instance of a cone_ in elementary geometry. *Circular* means that the base of the
        cone is a circle. *Right* means that the axis of the cone passes through the center of the base at
        right angles to its plane.

    Ruled surface
        A surface :math:`S` is ruled if through every point of :math:`S` there is a straight line that lies on
        :math:`S`. The plane and the lateral surface of a :term:`cone <right circular cone>` are examples of
        `ruled surfaces <ruled-surface_>`_.

    Segment
        A specifically defined part of a :term:`figure` (or of a whole :term:`maneuver`) in which certain
        particular points are detailed. For example, the first loop which makes the first half of the first
        complete figure eight in the "two consecutive overhead eight" maneuver is referred to as a segment.

    Small circle
        The nontrivial intersection of a plane and a :term:`sphere` such that the plane does not contain the
        center of the sphere.

    Sphere
        A solid bounded by a surface every point of which is equally distant from a fixed point.

References
==========

.. [#atan2] See the ``math.atan2`` function in Python `documentation <math-atan2_>`_.

.. [#atan] See the ``math.atan`` function in Python `documentation <math-atan_>`_.

.. _Algebra: https://en.wikipedia.org/wiki/Algebra

.. _Trigonometry: https://en.wikipedia.org/wiki/Trigonometry

.. _euclidean-geometry: https://en.wikipedia.org/wiki/Euclidean_geometry

.. _linear-algebra: https://en.wikipedia.org/wiki/Linear_algebra

.. _Python: https://www.python.org/

.. _math-atan: https://docs.python.org/3/library/math.html#math.atan

.. _math-atan2: https://docs.python.org/3/library/math.html#math.atan2

.. _ebook-todhunter: https://gutenberg.org/ebooks/19770

.. _cone: https://en.wikipedia.org/wiki/Cone

.. _ruled-surface: https://en.wikipedia.org/wiki/Ruled_surface

.. [#TODO-need-ref] TODO: NEED REFERENCE
