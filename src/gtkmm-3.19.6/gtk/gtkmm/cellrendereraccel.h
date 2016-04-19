// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_CELLRENDERERACCEL_H
#define _GTKMM_CELLRENDERERACCEL_H


#include <glibmm/ustring.h>
#include <sigc++/sigc++.h>

/*
 * Copyright (C) 2005 The gtkmm Development Team
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

#include <gtkmm/cellrenderertext.h>
#include <gtkmm/accelkey.h>


#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct _GtkCellRendererAccel GtkCellRendererAccel;
typedef struct _GtkCellRendererAccelClass GtkCellRendererAccelClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Gtk
{ class CellRendererAccel_Class; } // namespace Gtk
#endif //DOXYGEN_SHOULD_SKIP_THIS

namespace Gtk
{

/** @addtogroup gtkmmEnums gtkmm Enums and Flags */

/** 
 *  @var CellRendererAccelMode CELL_RENDERER_ACCEL_MODE_GTK
 * GTK+ accelerators mode.
 * 
 *  @var CellRendererAccelMode CELL_RENDERER_ACCEL_MODE_OTHER
 * Other accelerator mode.
 * 
 *  @enum CellRendererAccelMode
 * 
 * Determines if the edited accelerators are GTK+ accelerators. If
 * they are, consumed modifiers are suppressed, only accelerators
 * accepted by GTK+ are allowed, and the accelerators are rendered
 * in the same way as they are in menus.
 *
 * @ingroup gtkmmEnums
 */
enum CellRendererAccelMode
{
  CELL_RENDERER_ACCEL_MODE_GTK,
  CELL_RENDERER_ACCEL_MODE_OTHER
};

} // namespace Gtk


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Glib
{

template <>
class Value<Gtk::CellRendererAccelMode> : public Glib::Value_Enum<Gtk::CellRendererAccelMode>
{
public:
  static GType value_type() G_GNUC_CONST;
};

} // namespace Glib
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


namespace Gtk
{


/**
 * Renders a keyboard accelerator in a cell.
 *
 * Gtk::CellRendererAccel displays a keyboard accelerator
 * (i.e. a key combination like <Control>-a).
 * If the cell renderer is editable, the accelerator can be changed by
 * simply typing the new combination.
 *
 * @ingroup TreeView
 * @newin{2,10}
 */

class CellRendererAccel : public CellRendererText
{
  public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef CellRendererAccel CppObjectType;
  typedef CellRendererAccel_Class CppClassType;
  typedef GtkCellRendererAccel BaseObjectType;
  typedef GtkCellRendererAccelClass BaseClassType;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  CellRendererAccel(CellRendererAccel&& src) noexcept;
  CellRendererAccel& operator=(CellRendererAccel&& src) noexcept;

  // noncopyable
  CellRendererAccel(const CellRendererAccel&) = delete;
  CellRendererAccel& operator=(const CellRendererAccel&) = delete;

  ~CellRendererAccel() noexcept override;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

private:
  friend class CellRendererAccel_Class;
  static CppClassType cellrendereraccel_class_;

protected:
  explicit CellRendererAccel(const Glib::ConstructParams& construct_params);
  explicit CellRendererAccel(GtkCellRendererAccel* castitem);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

public:

  /** Get the GType for this class, for use with the underlying GObject type system.
   */
  static GType get_type()      G_GNUC_CONST;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


  static GType get_base_type() G_GNUC_CONST;
#endif

  ///Provides access to the underlying C GtkObject.
  GtkCellRendererAccel*       gobj()       { return reinterpret_cast<GtkCellRendererAccel*>(gobject_); }

  ///Provides access to the underlying C GtkObject.
  const GtkCellRendererAccel* gobj() const { return reinterpret_cast<GtkCellRendererAccel*>(gobject_); }


public:
  //C++ methods used to invoke GTK+ virtual functions:

protected:
  //GTK+ Virtual Functions (override these to change behaviour):

  //Default Signal Handlers::
  /// This is a default handler for the signal signal_accel_edited().
  virtual void on_accel_edited(const Glib::ustring& path_string, guint accel_key, Gdk::ModifierType accel_mods, guint hardware_keycode);
  /// This is a default handler for the signal signal_accel_cleared().
  virtual void on_accel_cleared(const Glib::ustring& path_string);


private:

public:

  CellRendererAccel();

 
  //TODO: Wrap accel_key and accel_mods in an AccelKey?
  
  /**
   * @par Slot Prototype:
   * <tt>void on_my_%accel_edited(const Glib::ustring& path_string, guint accel_key, Gdk::ModifierType accel_mods, guint hardware_keycode)</tt>
   *
   * Gets emitted when the user has selected a new accelerator.
   * 
   * @newin{2,10}
   * 
   * @param path_string The path identifying the row of the edited cell.
   * @param accel_key The new accelerator keyval.
   * @param accel_mods The new acclerator modifier mask.
   * @param hardware_keycode The keycode of the new accelerator.
   */

  Glib::SignalProxy4< void,const Glib::ustring&,guint,Gdk::ModifierType,guint > signal_accel_edited();

  
  /**
   * @par Slot Prototype:
   * <tt>void on_my_%accel_cleared(const Glib::ustring& path_string)</tt>
   *
   * Gets emitted when the user has removed the accelerator.
   * 
   * @newin{2,10}
   * 
   * @param path_string The path identifying the row of the edited cell.
   */

  Glib::SignalProxy1< void,const Glib::ustring& > signal_accel_cleared();


  /** The keyval of the accelerator.
   * 
   * @newin{2,10}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< guint > property_accel_key() ;

/** The keyval of the accelerator.
   * 
   * @newin{2,10}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< guint > property_accel_key() const;

  /** The modifier mask of the accelerator.
   * 
   * @newin{2,10}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Gdk::ModifierType > property_accel_mods() ;

/** The modifier mask of the accelerator.
   * 
   * @newin{2,10}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Gdk::ModifierType > property_accel_mods() const;

  /** The hardware keycode of the accelerator. Note that the hardware keycode is
   * only relevant if the key does not have a keyval. Normally, the keyboard
   * configuration should assign keyvals to all keys.
   * 
   * @newin{2,10}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< guint > property_keycode() ;

/** The hardware keycode of the accelerator. Note that the hardware keycode is
   * only relevant if the key does not have a keyval. Normally, the keyboard
   * configuration should assign keyvals to all keys.
   * 
   * @newin{2,10}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< guint > property_keycode() const;

  /** Determines if the edited accelerators are GTK+ accelerators. If
   * they are, consumed modifiers are suppressed, only accelerators
   * accepted by GTK+ are allowed, and the accelerators are rendered
   * in the same way as they are in menus.
   * 
   * @newin{2,10}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< CellRendererAccelMode > property_accel_mode() ;

/** Determines if the edited accelerators are GTK+ accelerators. If
   * they are, consumed modifiers are suppressed, only accelerators
   * accepted by GTK+ are allowed, and the accelerators are rendered
   * in the same way as they are in menus.
   * 
   * @newin{2,10}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< CellRendererAccelMode > property_accel_mode() const;


  Glib::PropertyProxy_Base _property_renderable() override;


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
   * @relates Gtk::CellRendererAccel
   */
  Gtk::CellRendererAccel* wrap(GtkCellRendererAccel* object, bool take_copy = false);
} //namespace Glib


#endif /* _GTKMM_CELLRENDERERACCEL_H */

