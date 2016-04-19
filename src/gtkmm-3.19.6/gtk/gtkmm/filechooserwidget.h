// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_FILECHOOSERWIDGET_H
#define _GTKMM_FILECHOOSERWIDGET_H


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

#include <gtkmm/box.h>
#include <gtkmm/filechooser.h>


#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct _GtkFileChooserWidget GtkFileChooserWidget;
typedef struct _GtkFileChooserWidgetClass GtkFileChooserWidgetClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Gtk
{ class FileChooserWidget_Class; } // namespace Gtk
#endif //DOXYGEN_SHOULD_SKIP_THIS

namespace Gtk
{

/** File chooser widget that can be embedded in other widgets.
 *
 * FileChooserWidget is a widget suitable for selecting files. It is the main
 * building block of a Gtk::FileChooserDialog. Most applications will only need to use
 * the latter; you can use FileChooserWidget as part of a larger window if you have
 * special needs.
 *
 * @ingroup Widgets
 */

class FileChooserWidget
: public VBox,
  public FileChooser
{
  public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef FileChooserWidget CppObjectType;
  typedef FileChooserWidget_Class CppClassType;
  typedef GtkFileChooserWidget BaseObjectType;
  typedef GtkFileChooserWidgetClass BaseClassType;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  FileChooserWidget(FileChooserWidget&& src) noexcept;
  FileChooserWidget& operator=(FileChooserWidget&& src) noexcept;

  // noncopyable
  FileChooserWidget(const FileChooserWidget&) = delete;
  FileChooserWidget& operator=(const FileChooserWidget&) = delete;

  ~FileChooserWidget() noexcept override;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

private:
  friend class FileChooserWidget_Class;
  static CppClassType filechooserwidget_class_;

protected:
  explicit FileChooserWidget(const Glib::ConstructParams& construct_params);
  explicit FileChooserWidget(GtkFileChooserWidget* castitem);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

public:

  /** Get the GType for this class, for use with the underlying GObject type system.
   */
  static GType get_type()      G_GNUC_CONST;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


  static GType get_base_type() G_GNUC_CONST;
#endif

  ///Provides access to the underlying C GtkObject.
  GtkFileChooserWidget*       gobj()       { return reinterpret_cast<GtkFileChooserWidget*>(gobject_); }

  ///Provides access to the underlying C GtkObject.
  const GtkFileChooserWidget* gobj() const { return reinterpret_cast<GtkFileChooserWidget*>(gobject_); }


public:
  //C++ methods used to invoke GTK+ virtual functions:

protected:
  //GTK+ Virtual Functions (override these to change behaviour):

  //Default Signal Handlers::


private:

  
public:
  FileChooserWidget();

  /** Creates a file chooser widget that can be embedded in other widgets.
   *
   * Creates a new FileChooserWidget. This is a file chooser widget that can be embedded in
   * custom windows, and it is the same widget that is used by Gtk::FileChooserDialog.
   *
   * @param action Open or save mode for the widget
   *
   * @newin{2,4}
   */
    explicit FileChooserWidget(FileChooserAction action);


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
   * @relates Gtk::FileChooserWidget
   */
  Gtk::FileChooserWidget* wrap(GtkFileChooserWidget* object, bool take_copy = false);
} //namespace Glib


#endif /* _GTKMM_FILECHOOSERWIDGET_H */

