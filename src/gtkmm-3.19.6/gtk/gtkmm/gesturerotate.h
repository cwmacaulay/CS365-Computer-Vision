// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_GESTUREROTATE_H
#define _GTKMM_GESTUREROTATE_H


#include <glibmm/ustring.h>
#include <sigc++/sigc++.h>

/* Copyright (C) 2014 The gtkmm Development Team
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library. If not, see <http://www.gnu.org/licenses/>.
 */

#include <gtkmm/gesture.h>


#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct _GtkGestureRotate GtkGestureRotate;
typedef struct _GtkGestureRotateClass GtkGestureRotateClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Gtk
{ class GestureRotate_Class; } // namespace Gtk
#endif //DOXYGEN_SHOULD_SKIP_THIS

namespace Gtk
{
/** Rotate gesture.
 *
 * This is a Gesture implementation able to recognize
 * 2-finger rotations. Whenever the angle between both handled sequences
 * changes, signal_angle_changed() is emitted.
 *
 * @newin{3,14}
 *
 * @ingroup Gestures
 */

class GestureRotate : public Gesture
{
  
#ifndef DOXYGEN_SHOULD_SKIP_THIS

public:
  typedef GestureRotate CppObjectType;
  typedef GestureRotate_Class CppClassType;
  typedef GtkGestureRotate BaseObjectType;
  typedef GtkGestureRotateClass BaseClassType;

  // noncopyable
  GestureRotate(const GestureRotate&) = delete;
  GestureRotate& operator=(const GestureRotate&) = delete;

private:  friend class GestureRotate_Class;
  static CppClassType gesturerotate_class_;

protected:
  explicit GestureRotate(const Glib::ConstructParams& construct_params);
  explicit GestureRotate(GtkGestureRotate* castitem);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

public:

  GestureRotate(GestureRotate&& src) noexcept;
  GestureRotate& operator=(GestureRotate&& src) noexcept;

  ~GestureRotate() noexcept override;

  /** Get the GType for this class, for use with the underlying GObject type system.
   */
  static GType get_type()      G_GNUC_CONST;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


  static GType get_base_type() G_GNUC_CONST;
#endif

  ///Provides access to the underlying C GObject.
  GtkGestureRotate*       gobj()       { return reinterpret_cast<GtkGestureRotate*>(gobject_); }

  ///Provides access to the underlying C GObject.
  const GtkGestureRotate* gobj() const { return reinterpret_cast<GtkGestureRotate*>(gobject_); }

  ///Provides access to the underlying C instance. The caller is responsible for unrefing it. Use when directly setting fields in structs.
  GtkGestureRotate* gobj_copy();

private:


protected:
  /** There is no create() method that corresponds to this constructor,
   * because this constructor shall only be used by derived classes.
   */
  GestureRotate();

  /** Constructs a Gesture that recognizes 2-touch rotation gestures.
   */
    explicit GestureRotate(Widget& widget);


public:
  /** Creates a Gesture that recognizes 2-touch rotation gestures.
   *
   * @newin{3,14}
   *
   * @param widget Widget the gesture relates to.
   * @return A RefPtr to a new GestureRotate.
   */
  
  static Glib::RefPtr<GestureRotate> create(Widget& widget);


  /** If @a gesture is active, this function returns the angle difference
   * in radians since the gesture was first recognized. If @a gesture is
   * not active, 0 is returned.
   * 
   * @newin{3,14}
   * 
   * @return The angle delta in radians.
   */
  double get_angle_delta() const;

  // no_default_handler because GtkGestureRotateClass is private.
  
  /**
   * @par Slot Prototype:
   * <tt>void on_my_%angle_changed(double angle, double angle_delta)</tt>
   *
   * This signal is emitted when the angle between both tracked points
   * changes.
   * 
   * @newin{3,14}
   * 
   * @param angle Current angle in radians.
   * @param angle_delta Difference with the starting angle, in radians.
   */

  Glib::SignalProxy2< void,double,double > signal_angle_changed();


  // GestureRotate has no properties


public:

public:
  //C++ methods used to invoke GTK+ virtual functions:

protected:
  //GTK+ Virtual Functions (override these to change behaviour):

  //Default Signal Handlers::


};

} // namespace Gtk


namespace Glib
{
  /** A Glib::wrap() method for this object.
   * 
   * @param object The C instance.
   * @param take_copy False if the result should take ownership of the C instance. True if it should take a new copy or ref.
   * @result A C++ instance that wraps this C instance.
   *
   * @relates Gtk::GestureRotate
   */
  Glib::RefPtr<Gtk::GestureRotate> wrap(GtkGestureRotate* object, bool take_copy = false);
}


#endif /* _GTKMM_GESTUREROTATE_H */

