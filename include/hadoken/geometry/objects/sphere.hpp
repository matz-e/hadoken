/**
 * Copyright (c) 2016, Adrien Devresse <adrien.devresse@epfl.ch>
 *
 * Boost Software License - Version 1.0
 *
 * Permission is hereby granted, free of charge, to any person or organization
 * obtaining a copy of the software and accompanying documentation covered by
 * this license (the "Software") to use, reproduce, display, distribute,
 * execute, and transmit the Software, and to prepare derivative works of the
 * Software, and to permit third-parties to whom the Software is furnished to
 * do so, all subject to the following:
 *
 * The copyright notices in the Software and this entire statement, including
 * the above license grant, this restriction and the following disclaimer,
 * must be included in all copies of the Software, in whole or in part, and
 * all derivative works of the Software, unless such copies or derivative
 * works are solely in the form of machine-executable object code generated by
 * a source language processor.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
*
*/
#ifndef _HADOKEN_OBJECT_SPHERE_HPP
#define _HADOKEN_OBJECT_SPHERE_HPP


#include <boost/geometry.hpp>

namespace hadoken {

namespace geometry {

namespace cartesian {


struct sphere_object_tag {};
struct circle_object_tag {};

template<typename Point, typename CoordType>
class sphere_traits{
public:

    typedef Point point_type;
    typedef CoordType coordinate_type;

    inline sphere_traits(const point_type & center, coordinate_type radius): _center(center), _radius(radius) {}

    inline coordinate_type get_radius() const{
        return _radius;
    }

    inline point_type get_center() const{
        return _center;
    }

protected:
    point_type _center;
    coordinate_type _radius;
};


///
/// \class sphere_base
///
///  sphere object in a 3D cartesian space
///
template<typename Point, typename CoordType>
class sphere_base: public sphere_traits<Point, CoordType>{
public:
    sphere_object_tag object_type;

    sphere_base(const Point & center, CoordType radius): sphere_traits<Point, CoordType>(center, radius) {}

private:
};


///
/// \class circle_base
///
///  oriented circle object in a 3D cartesian space
///
template<typename Point, typename Vector, typename CoordType>
class circle_base: public sphere_traits<Point, CoordType>{
public:
    circle_object_tag object_type;
    typedef Vector vector_type;

    inline circle_base(const Point & center, CoordType radius, const Vector & axis) :
        sphere_traits<Point, CoordType>(center, radius),
        _axis(axis) {}

    inline vector_type get_axis() const{
        return _axis;
    }

private:
    vector_type _axis;
};


} // cartesian

} // geometry

} // hadoken

#endif // SPHERE_HPP
