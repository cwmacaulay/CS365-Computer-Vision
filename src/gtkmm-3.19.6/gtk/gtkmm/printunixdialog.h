// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!
#ifndef _GTKMM_PRINTUNIXDIALOG_H
#define _GTKMM_PRINTUNIXDIALOG_H


#include <glibmm/ustring.h>
#include <sigc++/sigc++.h>

/* Copyright (C) 2006 The gtkmm Development Team
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

#include <gtkmm/dialog.h>
#include <gtkmm/pagesetup.h>
#include <gtkmm/printer.h>
#include <gtkmm/printjob.h>
#include <gtkmm/printsettings.h>


#ifndef DOXYGEN_SHOULD_SKIP_THIS
typedef struct _GtkPrintUnixDialog GtkPrintUnixDialog;
typedef struct _GtkPrintUnixDialogClass GtkPrintUnixDialogClass;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */


#ifndef DOXYGEN_SHOULD_SKIP_THIS
namespace Gtk
{ class PrintUnixDialog_Class; } // namespace Gtk
#endif //DOXYGEN_SHOULD_SKIP_THIS

namespace Gtk
{

/** PrintUnixDialog implements a print dialog for platforms
 * which don't provide a native print dialog, like Unix. It can
 * be used very much like any other GTK+ dialog, at the cost of
 * the portability offered by the high-level printing API exposed
 * through PrintOperation.
 *
 * In order to print something with PrintUnixDialog, you need
 * to use get_selected_printer() to obtain a Printer object and
 * use it to construct a PrintJob using PrintJob::create().
 *
 * PrintUnixDialog uses the following response values:
 * - RESPONSE_OK for the "Print" button
 * - RESPONSE_APPLY for the "Preview" button
 * - RESPONSE_CANCEL for the "Cancel" button
 *
 * @newin{2,10}
 *
 * @ingroup Printing
 */

class PrintUnixDialog : public Dialog
{
  public:
#ifndef DOXYGEN_SHOULD_SKIP_THIS
  typedef PrintUnixDialog CppObjectType;
  typedef PrintUnixDialog_Class CppClassType;
  typedef GtkPrintUnixDialog BaseObjectType;
  typedef GtkPrintUnixDialogClass BaseClassType;
#endif /* DOXYGEN_SHOULD_SKIP_THIS */

  PrintUnixDialog(PrintUnixDialog&& src) noexcept;
  PrintUnixDialog& operator=(PrintUnixDialog&& src) noexcept;

  // noncopyable
  PrintUnixDialog(const PrintUnixDialog&) = delete;
  PrintUnixDialog& operator=(const PrintUnixDialog&) = delete;

  ~PrintUnixDialog() noexcept override;

#ifndef DOXYGEN_SHOULD_SKIP_THIS

private:
  friend class PrintUnixDialog_Class;
  static CppClassType printunixdialog_class_;

protected:
  explicit PrintUnixDialog(const Glib::ConstructParams& construct_params);
  explicit PrintUnixDialog(GtkPrintUnixDialog* castitem);

#endif /* DOXYGEN_SHOULD_SKIP_THIS */

public:

  /** Get the GType for this class, for use with the underlying GObject type system.
   */
  static GType get_type()      G_GNUC_CONST;

#ifndef DOXYGEN_SHOULD_SKIP_THIS


  static GType get_base_type() G_GNUC_CONST;
#endif

  ///Provides access to the underlying C GtkObject.
  GtkPrintUnixDialog*       gobj()       { return reinterpret_cast<GtkPrintUnixDialog*>(gobject_); }

  ///Provides access to the underlying C GtkObject.
  const GtkPrintUnixDialog* gobj() const { return reinterpret_cast<GtkPrintUnixDialog*>(gobject_); }


public:
  //C++ methods used to invoke GTK+ virtual functions:

protected:
  //GTK+ Virtual Functions (override these to change behaviour):

  //Default Signal Handlers::


private:

  
  //This is not available on Win32.
//This source file will not be compiled on Win32,
//and no class defined in it will be registered by wrap_init().


public:
  explicit PrintUnixDialog(Gtk::Window& parent, const Glib::ustring& title);
  explicit PrintUnixDialog(const Glib::ustring& title);

  
  /** Sets the page setup of the Gtk::PrintUnixDialog.
   * 
   * @newin{2,10}
   * 
   * @param page_setup A Gtk::PageSetup.
   */
  void set_page_setup(const Glib::RefPtr<PageSetup>& page_setup);
  
  /** Gets the page setup that is used by the Gtk::PrintUnixDialog.
   * 
   * @newin{2,10}
   * 
   * @return The page setup of @a dialog.
   */
  Glib::RefPtr<PageSetup> get_page_setup();
  
  /** Gets the page setup that is used by the Gtk::PrintUnixDialog.
   * 
   * @newin{2,10}
   * 
   * @return The page setup of @a dialog.
   */
  Glib::RefPtr<const PageSetup> get_page_setup() const;

  
  /** Sets the current page number. If @a current_page is not -1, this enables
   * the current page choice for the range of pages to print.
   * 
   * @newin{2,10}
   * 
   * @param current_page The current page number.
   */
  void set_current_page(int current_page);
  
  /** Gets the current page of the Gtk::PrintUnixDialog.
   * 
   * @newin{2,10}
   * 
   * @return The current page of @a dialog.
   */
  int get_current_page() const;

  
  /** Sets the Gtk::PrintSettings for the Gtk::PrintUnixDialog. Typically,
   * this is used to restore saved print settings from a previous print
   * operation before the print dialog is shown.
   * 
   * @newin{2,10}
   * 
   * @param settings A Gtk::PrintSettings, or <tt>nullptr</tt>.
   */
  void set_settings(const Glib::RefPtr<PrintSettings>& settings);
  
  /** Gets a new Gtk::PrintSettings object that represents the
   * current values in the print dialog. Note that this creates a
   * new object, and you need to unref it
   * if don’t want to keep it.
   * 
   * @newin{2,10}
   * 
   * @return A new Gtk::PrintSettings object with the values from @a dialog.
   */
  Glib::RefPtr<PrintSettings> get_settings();
  
  /** Gets a new Gtk::PrintSettings object that represents the
   * current values in the print dialog. Note that this creates a
   * new object, and you need to unref it
   * if don’t want to keep it.
   * 
   * @newin{2,10}
   * 
   * @return A new Gtk::PrintSettings object with the values from @a dialog.
   */
  Glib::RefPtr<const PrintSettings> get_settings() const;

   
  /** Gets the currently selected printer.
   * 
   * @newin{2,10}
   * 
   * @return The currently selected printer.
   */
  Glib::RefPtr<Printer> get_selected_printer();
  
  /** Gets the currently selected printer.
   * 
   * @newin{2,10}
   * 
   * @return The currently selected printer.
   */
  Glib::RefPtr<const Printer> get_selected_printer() const;

  void add_custom_tab(const Widget& child, const Glib::ustring& tab_label);
  
  /** Adds a custom tab to the print dialog.
   * 
   * @newin{2,10}
   * 
   * @param child The widget to put in the custom tab.
   * @param tab_label The widget to use as tab label.
   */
  void add_custom_tab(const Widget& child, const Widget& tab_label);
  
  /** This lets you specify the printing capabilities your application
   * supports. For instance, if you can handle scaling the output then
   * you pass Gtk::PRINT_CAPABILITY_SCALE. If you don’t pass that, then
   * the dialog will only let you select the scale if the printing
   * system automatically handles scaling.
   * 
   * @newin{2,10}
   * 
   * @param capabilities The printing capabilities of your application.
   */
  void set_manual_capabilities(PrintCapabilities capabilities);
  
  /** Gets the value of Gtk::PrintUnixDialog::property_manual_capabilities() property.
   * 
   * @newin{2,18}
   * 
   * @return The printing capabilities.
   */
  PrintCapabilities get_manual_capabilities() const;

  
  /** Sets whether the print dialog allows user to print a selection.
   * 
   * @newin{2,18}
   * 
   * @param support_selection <tt>true</tt> to allow print selection.
   */
  void get_support_selection(bool support_selection =  true);
  
  /** Gets the value of Gtk::PrintUnixDialog::property_support_selection() property.
   * 
   * @newin{2,18}
   * 
   * @return Whether the application supports print of selection.
   */
  bool get_support_selection() const;
  
  /** Sets whether a selection exists.
   * 
   * @newin{2,18}
   * 
   * @param has_selection <tt>true</tt> indicates that a selection exists.
   */
  void set_has_selection(bool has_selection =  true);
  
  /** Gets the value of Gtk::PrintUnixDialog::property_has_selection() property.
   * 
   * @newin{2,18}
   * 
   * @return Whether there is a selection.
   */
  bool get_has_selection() const;
  
  /** Embed page size combo box and orientation combo box into page setup page.
   * 
   * @newin{2,18}
   * 
   * @param embed Embed page setup selection.
   */
  void set_embed_page_setup(bool embed =  true);
  
  /** Gets the value of Gtk::PrintUnixDialog::property_embed_page_setup() property.
   * 
   * @newin{2,18}
   * 
   * @return Whether there is a selection.
   */
  bool get_embed_page_setup() const;
  
  /** Gets the page setup that is used by the Gtk::PrintUnixDialog.
   * 
   * @newin{2,18}
   * 
   * @return Whether a page setup was set by user.
   */
  bool get_page_setup_set() const;

  /** The GtkPageSetup to use.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::RefPtr<PageSetup> > property_page_setup() ;

/** The GtkPageSetup to use.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<PageSetup> > property_page_setup() const;

  /** The current page in the document.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< int > property_current_page() ;

/** The current page in the document.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< int > property_current_page() const;

  /** The GtkPrintSettings used for initializing the dialog.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< Glib::RefPtr<PrintSettings> > property_print_settings() ;

/** The GtkPrintSettings used for initializing the dialog.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<PrintSettings> > property_print_settings() const;

  /** The GtkPrinter which is selected.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< Glib::RefPtr<Printer> > property_selected_printer() const;


  /** Capabilities the application can handle.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< bool > property_manual_capabilities() ;

/** Capabilities the application can handle.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< bool > property_manual_capabilities() const;

  /** Whether the dialog supports selection.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< bool > property_support_selection() ;

/** Whether the dialog supports selection.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< bool > property_support_selection() const;

  /** Whether the application has a selection.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< bool > property_has_selection() ;

/** Whether the application has a selection.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< bool > property_has_selection() const;

  /** TRUE if page setup combos are embedded in GtkPrintUnixDialog.
   *
   * @return A PropertyProxy that allows you to get or set the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy< bool > property_embed_page_setup() ;

/** TRUE if page setup combos are embedded in GtkPrintUnixDialog.
   *
   * @return A PropertyProxy_ReadOnly that allows you to get the value of the property,
   * or receive notification when the value of the property changes.
   */
  Glib::PropertyProxy_ReadOnly< bool > property_embed_page_setup() const;


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
   * @relates Gtk::PrintUnixDialog
   */
  Gtk::PrintUnixDialog* wrap(GtkPrintUnixDialog* object, bool take_copy = false);
} //namespace Glib


#endif /* _GTKMM_PRINTUNIXDIALOG_H */
