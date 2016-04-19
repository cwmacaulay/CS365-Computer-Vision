// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_CELLRENDERERPIXBUF_H
#define _GTKMM_CELLRENDERERPIXBUF_H


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

#include <gtkmm/cellrenderer.h>
#include <giomm/icon.h>


#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct _GtkCellRendererPixbuf GtkCellRendererPixbuf;
typedef struct _GtkCellRendererPixbufClass GtkCellRendererPixbufClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Gtk
{ class CellRendererPixbuf_Class; } // namespace Gtk
#endif //DOXYGEN_SHOULD_SKIP_THIS

namespace Gtk
{

/** Renders a pixbuf in a cell.
 * A CellRendererPixbuf can be used to render an image in a cell. It allows you to render either a
 * given Pixbuf (set via the pixbuf property) or a stock icon (set via the stock_id property).
 *
 * To support the tree view, CellRendererPixbuf also supports rendering two alternative pixbufs, when
 * the is_expanded property is true. If the is_expanded property is true and the pixbuf_expander_open
 * property is set to a pixbuf, it renders that pixbuf. If the is_expanded property is false and the *
 * pixbuf_expander_closed property is set to a pixbuf, it renders that one.
 *
 * @ingroup TreeView
 */

class CellRendererPixbuf : public CellRenderer
{
  public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef CellRendererPixbuf CppObjectType;
  typedef CellRendererPixbuf_Class CppClassType;
  typedef GtkCellRendererPixbuf BaseObjectType;
  typedef GtkCellRendererPixbufClass BaseClassType;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  CellRendererPixbuf(CellRendererPixbuf&& src) noexcept;
  CellRendererPixbuf& operator=(CellRendererPixbuf&& src) noexcept;

  // noncopyable
  CellRendererPixbuf(const CellRendererPixbuf&) = delete;
  CellRendererPixbuf& operator=(const CellRendererPixbuf&) = delete;

  ~CellRendererPixbuf() noexcept override;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

private:
  friend class CellRendererPixbuf_Class;
  static CppClassType cellrendererpixbuf_class_;

protected:
  explicit CellRendererPixbuf(const Glib::ConstructParams& construct_params);
  explicit CellRendererPixbuf(GtkCellRendererPixbuf* castitem);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

public:

  /** Get the GType for this class, for use with the underlying GObject type system.
   */
  static GType get_type()      G_GNUC_CONST;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


  static GType get_base_type() G_GNUC_CONST;
#endif

  ///Provides access to the underlying C GtkObject.
  GtkCellRendererPixbuf*       gobj()       { return reinterpret_cast<GtkCellRendererPixbuf*>(gobject_); }

  ///Provides access to the underlying C GtkObject.
  const GtkCellRendererPixbuf* gobj() const { return reinterpret_cast<GtkCellRendererPixbuf*>(gobject_); }


public:
  //C++ methods used to invoke GTK+ virtual functions:

protected:
  //GTK+ Virtual Functions (override these to change behaviour):

  //Default Signal Handlers::


private:

public:

  CellRendererPixbuf();

  /** The pixbuf to render.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::RefPtr<Gdk::Pixbuf> > property_pixbuf() ;

/** The pixbuf to render.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<Gdk::Pixbuf> > property_pixbuf() const;

  /** Pixbuf for open expander.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::RefPtr<Gdk::Pixbuf> > property_pixbuf_expander_open() ;

/** Pixbuf for open expander.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<Gdk::Pixbuf> > property_pixbuf_expander_open() const;

  /** Pixbuf for closed expander.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::RefPtr<Gdk::Pixbuf> > property_pixbuf_expander_closed() ;

/** Pixbuf for closed expander.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<Gdk::Pixbuf> > property_pixbuf_expander_closed() const;

  
#ifndef GTKMM_DISABLE_DEPRECATED

/** The stock ID of the stock icon to render.
   *
   * Deprecated: 3.10: Use Gtk::CellRendererPixbuf::property_icon_name() instead.
   * 
   * @deprecated Use property_icon_name() instead.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::ustring > property_stock_id() ;

/** The stock ID of the stock icon to render.
   *
   * Deprecated: 3.10: Use Gtk::CellRendererPixbuf::property_icon_name() instead.
   * 
   * @deprecated Use property_icon_name() instead.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::ustring > property_stock_id() const;

#endif // GTKMM_DISABLE_DEPRECATED

  /** The GtkIconSize value that specifies the size of the rendered icon.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< guint > property_stock_size() ;

/** The GtkIconSize value that specifies the size of the rendered icon.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< guint > property_stock_size() const;

  /** Render detail to pass to the theme engine.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::ustring > property_stock_detail() ;

/** Render detail to pass to the theme engine.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::ustring > property_stock_detail() const;

  /** The name of the themed icon to display.
   * This property only has an effect if not overridden by "stock_id" 
   * or "pixbuf" properties.
   * 
   * @newin{2,8}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::ustring > property_icon_name() ;

/** The name of the themed icon to display.
   * This property only has an effect if not overridden by "stock_id" 
   * or "pixbuf" properties.
   * 
   * @newin{2,8}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::ustring > property_icon_name() const;

  
#ifndef GTKMM_DISABLE_DEPRECATED

/** Specifies whether the rendered pixbuf should be colorized
   * according to the Gtk::CellRendererState.
   * 
   * @newin{2,8}
   * 
   * Deprecated: 3.16: Cell renderers always follow state.
   * 
   * @deprecated Cell renderers always follow state.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< bool > property_follow_state() ;

/** Specifies whether the rendered pixbuf should be colorized
   * according to the Gtk::CellRendererState.
   * 
   * @newin{2,8}
   * 
   * Deprecated: 3.16: Cell renderers always follow state.
   * 
   * @deprecated Cell renderers always follow state.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< bool > property_follow_state() const;

#endif // GTKMM_DISABLE_DEPRECATED

  /** The GIcon representing the icon to display.
   * If the icon theme is changed, the image will be updated
   * automatically.
   * 
   * @newin{2,14}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::RefPtr<Gio::Icon> > property_gicon() ;

/** The GIcon representing the icon to display.
   * If the icon theme is changed, the image will be updated
   * automatically.
   * 
   * @newin{2,14}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<Gio::Icon> > property_gicon() const;

  /** The surface to render.
   *
   * @newin{3,10}
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< ::Cairo::RefPtr< ::Cairo::Surface> > property_surface() ;

/** The surface to render.
   *
   * @newin{3,10}
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< ::Cairo::RefPtr< ::Cairo::Surface> > property_surface() const;


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
   * @relates Gtk::CellRendererPixbuf
   */
  Gtk::CellRendererPixbuf* wrap(GtkCellRendererPixbuf* object, bool take_copy = false);
} //namespace Glib


#endif /* _GTKMM_CELLRENDERERPIXBUF_H */

