// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_HVSCROLLBAR_H
#define _GTKMM_HVSCROLLBAR_H


#ifndef GTKMM_DISABLE_DEPRECATED


#include <glibmm/ustring.h>
#include <sigc++/sigc++.h>

/*
 * Copyright (C) 1998-2002 The gtkmm Development Team
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
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include <gtkmm/scrollbar.h>


#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct _GtkVScrollbar GtkVScrollbar;
typedef struct _GtkVScrollbarClass GtkVScrollbarClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Gtk
{ class VScrollbar_Class; } // namespace Gtk
#endif //DOXYGEN_SHOULD_SKIP_THIS

#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct _GtkHScrollbar GtkHScrollbar;
typedef struct _GtkHScrollbarClass GtkHScrollbarClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Gtk
{ class HScrollbar_Class; } // namespace Gtk
#endif //DOXYGEN_SHOULD_SKIP_THIS

namespace Gtk
{

/** A vertical scrollbar.
 *
 * The Gtk::VScrollbar widget is a widget arranged vertically creating a
 * scrollbar. See Gtk::Scrollbar for details on scrollbars.
 *
 * A Gtk::Adjustment may may be passed to the constructor  to handle the
 * adjustment of the scrollbar. If not specified, one will be created for
 * you. See Gtk::Adjustment for details.
 *
 * A VScrollbar widget looks like this:
 * @image html vscrollbar1.png
 *
 * @ingroup Widgets
 *
 * @deprecated Use Scrollbar instead.
 */

class VScrollbar : public Scrollbar
{
  public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef VScrollbar CppObjectType;
  typedef VScrollbar_Class CppClassType;
  typedef GtkVScrollbar BaseObjectType;
  typedef GtkVScrollbarClass BaseClassType;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  VScrollbar(VScrollbar&& src) noexcept;
  VScrollbar& operator=(VScrollbar&& src) noexcept;

  // noncopyable
  VScrollbar(const VScrollbar&) = delete;
  VScrollbar& operator=(const VScrollbar&) = delete;

  ~VScrollbar() noexcept override;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

private:
  friend class VScrollbar_Class;
  static CppClassType vscrollbar_class_;

protected:
  explicit VScrollbar(const Glib::ConstructParams& construct_params);
  explicit VScrollbar(GtkVScrollbar* castitem);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

public:

  /** Get the GType for this class, for use with the underlying GObject type system.
   */
  static GType get_type()      G_GNUC_CONST;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


  static GType get_base_type() G_GNUC_CONST;
#endif

  ///Provides access to the underlying C GtkObject.
  GtkVScrollbar*       gobj()       { return reinterpret_cast<GtkVScrollbar*>(gobject_); }

  ///Provides access to the underlying C GtkObject.
  const GtkVScrollbar* gobj() const { return reinterpret_cast<GtkVScrollbar*>(gobject_); }


public:
  //C++ methods used to invoke GTK+ virtual functions:

protected:
  //GTK+ Virtual Functions (override these to change behaviour):

  //Default Signal Handlers::


private:

  
public:
  VScrollbar();
  explicit VScrollbar(const Glib::RefPtr<Adjustment>& gtkadjustment);


};

/** A horizontal scrollbar.
 *
 * The Gtk::HScrollbar widget is a widget arranged horizontally creating a
 * scrollbar. See Gtk::Scrollbar for details on scrollbars.
 *
 * A Gtk::Adjustment may may be passed to the constructor  to handle the
 * adjustment of the scrollbar. If not specified, one will be created for
 * you. See Gtk::Adjustment for details.
 *
 * The HScrollbar widget looks like this:
 * @image html hscrollbar1.png
 *
 * @ingroup Widgets
 *
 * @deprecated Use Scrollbar instead.
 */

class HScrollbar : public Scrollbar
{
  public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef HScrollbar CppObjectType;
  typedef HScrollbar_Class CppClassType;
  typedef GtkHScrollbar BaseObjectType;
  typedef GtkHScrollbarClass BaseClassType;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  HScrollbar(HScrollbar&& src) noexcept;
  HScrollbar& operator=(HScrollbar&& src) noexcept;

  // noncopyable
  HScrollbar(const HScrollbar&) = delete;
  HScrollbar& operator=(const HScrollbar&) = delete;

  ~HScrollbar() noexcept override;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

private:
  friend class HScrollbar_Class;
  static CppClassType hscrollbar_class_;

protected:
  explicit HScrollbar(const Glib::ConstructParams& construct_params);
  explicit HScrollbar(GtkHScrollbar* castitem);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

public:

  /** Get the GType for this class, for use with the underlying GObject type system.
   */
  static GType get_type()      G_GNUC_CONST;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


  static GType get_base_type() G_GNUC_CONST;
#endif

  ///Provides access to the underlying C GtkObject.
  GtkHScrollbar*       gobj()       { return reinterpret_cast<GtkHScrollbar*>(gobject_); }

  ///Provides access to the underlying C GtkObject.
  const GtkHScrollbar* gobj() const { return reinterpret_cast<GtkHScrollbar*>(gobject_); }


public:
  //C++ methods used to invoke GTK+ virtual functions:

protected:
  //GTK+ Virtual Functions (override these to change behaviour):

  //Default Signal Handlers::


private:

  
public:
  HScrollbar();
  explicit HScrollbar(const Glib::RefPtr<Adjustment>& gtkadjustment);


};


} //namespace Gtk


namespace Glib
{
  /** A Glib::wrap() method for this object.
   * 
   * @param object The C instance.
   * @param take_copy False if the result should take ownership of the C instance. True if it should take a new copy or ref.
   * @result A C++ instance that wraps this C instance.
   *
   * @relates Gtk::VScrollbar
   */
  Gtk::VScrollbar* wrap(GtkVScrollbar* object, bool take_copy = false);
} //namespace Glib


namespace Glib
{
  /** A Glib::wrap() method for this object.
   * 
   * @param object The C instance.
   * @param take_copy False if the result should take ownership of the C instance. True if it should take a new copy or ref.
   * @result A C++ instance that wraps this C instance.
   *
   * @relates Gtk::HScrollbar
   */
  Gtk::HScrollbar* wrap(GtkHScrollbar* object, bool take_copy = false);
} //namespace Glib


#endif // GTKMM_DISABLE_DEPRECATED


#endif /* _GTKMM_HVSCROLLBAR_H */
