// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_TOOLITEMGROUP_H
#define _GTKMM_TOOLITEMGROUP_H


#include <glibmm/ustring.h>
#include <sigc++/sigc++.h>

/*
 * Copyright (C) 2009 The gtkmm Development Team
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


#include <gtkmm/container.h>
#include <gtkmm/toolitem.h>
#include <gtkmm/toolshell.h>


#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct _GtkToolItemGroup GtkToolItemGroup;
typedef struct _GtkToolItemGroupClass GtkToolItemGroupClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Gtk
{ class ToolItemGroup_Class; } // namespace Gtk
#endif //DOXYGEN_SHOULD_SKIP_THIS

namespace Gtk
{

/** A ToolItemGroup is used together with ToolPalette to add ToolItems to a
 * palette-like container with different categories and drag and drop support.
 *
 * @newin{2,20}
 * @ingroup Widgets
 * @ingroup Containers
 */

class ToolItemGroup
 : public Container,
   public ToolShell
{
  public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef ToolItemGroup CppObjectType;
  typedef ToolItemGroup_Class CppClassType;
  typedef GtkToolItemGroup BaseObjectType;
  typedef GtkToolItemGroupClass BaseClassType;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  ToolItemGroup(ToolItemGroup&& src) noexcept;
  ToolItemGroup& operator=(ToolItemGroup&& src) noexcept;

  // noncopyable
  ToolItemGroup(const ToolItemGroup&) = delete;
  ToolItemGroup& operator=(const ToolItemGroup&) = delete;

  ~ToolItemGroup() noexcept override;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

private:
  friend class ToolItemGroup_Class;
  static CppClassType toolitemgroup_class_;

protected:
  explicit ToolItemGroup(const Glib::ConstructParams& construct_params);
  explicit ToolItemGroup(GtkToolItemGroup* castitem);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

public:

  /** Get the GType for this class, for use with the underlying GObject type system.
   */
  static GType get_type()      G_GNUC_CONST;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


  static GType get_base_type() G_GNUC_CONST;
#endif

  ///Provides access to the underlying C GtkObject.
  GtkToolItemGroup*       gobj()       { return reinterpret_cast<GtkToolItemGroup*>(gobject_); }

  ///Provides access to the underlying C GtkObject.
  const GtkToolItemGroup* gobj() const { return reinterpret_cast<GtkToolItemGroup*>(gobject_); }


public:
  //C++ methods used to invoke GTK+ virtual functions:

protected:
  //GTK+ Virtual Functions (override these to change behaviour):

  //Default Signal Handlers::


private:

  
public:
    explicit ToolItemGroup(const Glib::ustring& label =  Glib::ustring());


  /** Sets the label of the tool item group. The label is displayed in the header
   * of the group.
   * 
   * @newin{2,20}
   * 
   * @param label The new human-readable label of of the group.
   */
  void set_label(const Glib::ustring& label);
  
  /** Sets the label of the tool item group.
   * The label widget is displayed in the header of the group, in place
   * of the usual label.
   * 
   * @newin{2,20}
   * 
   * @param label_widget The widget to be displayed in place of the usual label.
   */
  void set_label_widget(Widget& label_widget);
  
  /** Sets whether the @a group should be collapsed or expanded.
   * 
   * @newin{2,20}
   * 
   * @param collapsed Whether the @a group should be collapsed or expanded.
   */
  void set_collapsed(bool collapsed =  true);
  
  /** Sets the ellipsization mode which should be used by labels in @a group.
   * 
   * @newin{2,20}
   * 
   * @param ellipsize The Pango::EllipsizeMode labels in @a group should use.
   */
  void set_ellipsize(Pango::EllipsizeMode ellipsize);
  
  /** Set the button relief of the group header.
   * See Gtk::Button::set_relief() for details.
   * 
   * @newin{2,20}
   * 
   * @param style The Gtk::ReliefStyle.
   */
  void set_header_relief(ReliefStyle style);

  
  /** Gets the label of @a group.
   * 
   * @newin{2,20}
   * 
   * @return The label of @a group. The label is an internal string of @a group
   * and must not be modified. Note that <tt>nullptr</tt> is returned if a custom
   * label has been set with set_label_widget().
   */
  Glib::ustring get_label() const;

  
  /** Gets the label widget of @a group.
   * See set_label_widget().
   * 
   * @newin{2,20}
   * 
   * @return The label widget of @a group.
   */
  Widget* get_label_widget();
  
  /** Gets the label widget of @a group.
   * See set_label_widget().
   * 
   * @newin{2,20}
   * 
   * @return The label widget of @a group.
   */
  const Widget* get_label_widget() const;

  
  /** Gets whether @a group is collapsed or expanded.
   * 
   * @newin{2,20}
   * 
   * @return <tt>true</tt> if @a group is collapsed, <tt>false</tt> if it is expanded.
   */
  bool get_collapsed() const;
  
  /** Gets the ellipsization mode of @a group.
   * 
   * @newin{2,20}
   * 
   * @return The Pango::EllipsizeMode of @a group.
   */
  Pango::EllipsizeMode get_ellipsize() const;
  
  /** Gets the relief mode of the header button of @a group.
   * 
   * @newin{2,20}
   * 
   * @return The Gtk::ReliefStyle.
   */
  ReliefStyle get_header_relief() const;

  
  /** Inserts @a item at @a position in the list of children of @a group.
   * 
   * @newin{2,20}
   * 
   * @param item The Gtk::ToolItem to insert into @a group.
   * @param position The position of @a item in @a group, starting with 0.
   * The position -1 means end of list.
   */
  void insert(ToolItem& item, int position);

  /** Inserts @a item at the end of the list of children of the group.
   *
   * @param item The ToolItem to insert into the group.
   */
  void insert(ToolItem& item);

  
  /** Sets the position of @a item in the list of children of @a group.
   * 
   * @newin{2,20}
   * 
   * @param item The Gtk::ToolItem to move to a new position, should
   * be a child of @a group.
   * @param position The new position of @a item in @a group, starting with 0.
   * The position -1 means end of list.
   */
  void set_item_position(ToolItem& item, int position);
  
  /** Gets the position of @a item in @a group as index.
   * 
   * @newin{2,20}
   * 
   * @param item A Gtk::ToolItem.
   * @return The index of @a item in @a group or -1 if @a item is no child of @a group.
   */
  int get_item_position(const ToolItem& item) const;

  
  /** Gets the number of tool items in @a group.
   * 
   * @newin{2,20}
   * 
   * @return The number of tool items in @a group.
   */
  guint get_n_items() const;

  
  /** Gets the tool item at @a index in group.
   * 
   * @newin{2,20}
   * 
   * @param index The index.
   * @return The Gtk::ToolItem at index.
   */
  ToolItem* get_nth_item(guint index);
  
  /** Gets the tool item at @a index in group.
   * 
   * @newin{2,20}
   * 
   * @param index The index.
   * @return The Gtk::ToolItem at index.
   */
  const ToolItem* get_nth_item(guint index) const;

  
  /** Gets the tool item at position (x, y).
   * 
   * @newin{2,20}
   * 
   * @param x The x position.
   * @param y The y position.
   * @return The Gtk::ToolItem at position (x, y).
   */
  ToolItem* get_drop_item(int x, int y);
  
  /** Gets the tool item at position (x, y).
   * 
   * @newin{2,20}
   * 
   * @param x The x position.
   * @param y The y position.
   * @return The Gtk::ToolItem at position (x, y).
   */
  ToolItem* get_drop_item(int x, int y) const;

  /** The human-readable title of this item group.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::ustring > property_label() ;

/** The human-readable title of this item group.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::ustring > property_label() const;

  /** A widget to display in place of the usual label.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Gtk::Widget* > property_label_widget() ;

/** A widget to display in place of the usual label.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Gtk::Widget* > property_label_widget() const;

  /** Whether the group has been collapsed and items are hidden.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< bool > property_collapsed() ;

/** Whether the group has been collapsed and items are hidden.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< bool > property_collapsed() const;

  /** Ellipsize for item group headers.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Pango::EllipsizeMode > property_ellipsize() ;

/** Ellipsize for item group headers.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Pango::EllipsizeMode > property_ellipsize() const;

  /** Relief of the group header button.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< ReliefStyle > property_header_relief() ;

/** Relief of the group header button.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< ReliefStyle > property_header_relief() const;


  /** Whether the item should be the same size as other homogeneous items.
   *
   * @return A ChildPropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Gtk::ChildPropertyProxy< bool > child_property_homogeneous(Gtk::Widget& child) ;

/** Whether the item should be the same size as other homogeneous items.
   *
   * @return A ChildPropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Gtk::ChildPropertyProxy_ReadOnly< bool > child_property_homogeneous(const Gtk::Widget& child) const;

  /** Whether the item should receive extra space when the group grows.
   *
   * @return A ChildPropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Gtk::ChildPropertyProxy< bool > child_property_expand(Gtk::Widget& child) ;

/** Whether the item should receive extra space when the group grows.
   *
   * @return A ChildPropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Gtk::ChildPropertyProxy_ReadOnly< bool > child_property_expand(const Gtk::Widget& child) const;

  /** Whether the item should fill the available space.
   *
   * @return A ChildPropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Gtk::ChildPropertyProxy< bool > child_property_fill(Gtk::Widget& child) ;

/** Whether the item should fill the available space.
   *
   * @return A ChildPropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Gtk::ChildPropertyProxy_ReadOnly< bool > child_property_fill(const Gtk::Widget& child) const;

  /** Whether the item should start a new row.
   *
   * @return A ChildPropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Gtk::ChildPropertyProxy< bool > child_property_new_row(Gtk::Widget& child) ;

/** Whether the item should start a new row.
   *
   * @return A ChildPropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Gtk::ChildPropertyProxy_ReadOnly< bool > child_property_new_row(const Gtk::Widget& child) const;

  /** Position of the item within this group.
   *
   * @return A ChildPropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Gtk::ChildPropertyProxy< int > child_property_position(Gtk::Widget& child) ;

/** Position of the item within this group.
   *
   * @return A ChildPropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Gtk::ChildPropertyProxy_ReadOnly< int > child_property_position(const Gtk::Widget& child) const;


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
   * @relates Gtk::ToolItemGroup
   */
  Gtk::ToolItemGroup* wrap(GtkToolItemGroup* object, bool take_copy = false);
} //namespace Glib


#endif /* _GTKMM_TOOLITEMGROUP_H */

