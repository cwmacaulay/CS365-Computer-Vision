// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!

#undef GTK_DISABLE_DEPRECATED
#define GDK_DISABLE_DEPRECATION_WARNINGS 1
 

#ifndef GTKMM_DISABLE_DEPRECATED


#include <glibmm.h>

#include <gtkmm/fontselection.h>
#include <gtkmm/private/fontselection_p.h>


/* Copyright 1998-2002 The gtkmm Development Team
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

#include <gtkmm/button.h>
#include <gtkmm/entry.h>
#include <gtkmm/radiobutton.h>
#include <gtkmm/treeview.h>
#include <gtk/gtk.h>


namespace
{
} // anonymous namespace


namespace Glib
{

Gtk::FontSelection* wrap(GtkFontSelection* object, bool take_copy)
{
  return dynamic_cast<Gtk::FontSelection *> (Glib::wrap_auto ((GObject*)(object), take_copy));
}

} /* namespace Glib */

namespace Gtk
{


/* The *_Class implementation: */

const Glib::Class& FontSelection_Class::init()
{
  if(!gtype_) // create the GType if necessary
  {
    // Glib::Class has to know the class init function to clone custom types.
    class_init_func_ = &FontSelection_Class::class_init_function;

    // This is actually just optimized away, apparently with no harm.
    // Make sure that the parent type has been created.
    //CppClassParent::CppObjectType::get_type();

    // Create the wrapper type, with the same class/instance size as the base type.
    register_derived_type(gtk_font_selection_get_type());

    // Add derived versions of interfaces, if the C type implements any interfaces:

  }

  return *this;
}


void FontSelection_Class::class_init_function(void* g_class, void* class_data)
{
  const auto klass = static_cast<BaseClassType*>(g_class);
  CppClassParent::class_init_function(klass, class_data);


}


Glib::ObjectBase* FontSelection_Class::wrap_new(GObject* o)
{
  return manage(new FontSelection((GtkFontSelection*)(o)));

}


/* The implementation: */

FontSelection::FontSelection(const Glib::ConstructParams& construct_params)
:
  Gtk::VBox(construct_params)
{
  }

FontSelection::FontSelection(GtkFontSelection* castitem)
:
  Gtk::VBox((GtkVBox*)(castitem))
{
  }


FontSelection::FontSelection(FontSelection&& src) noexcept
: Gtk::VBox(std::move(src))
{}

FontSelection& FontSelection::operator=(FontSelection&& src) noexcept
{
  Gtk::VBox::operator=(std::move(src));
  return *this;
}

FontSelection::~FontSelection() noexcept
{
  destroy_();
}

FontSelection::CppClassType FontSelection::fontselection_class_; // initialize static member

GType FontSelection::get_type()
{
  return fontselection_class_.init().get_type();
}


GType FontSelection::get_base_type()
{
  return gtk_font_selection_get_type();
}


FontSelection::FontSelection()
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  Gtk::VBox(Glib::ConstructParams(fontselection_class_.init()))
{
  

}

TreeView* FontSelection::get_family_list()
{
  return Glib::wrap((GtkTreeView*)(gtk_font_selection_get_family_list(gobj())));
}

const TreeView* FontSelection::get_family_list() const
{
  return const_cast<FontSelection*>(this)->get_family_list();
}

TreeView* FontSelection::get_face_list()
{
  return Glib::wrap((GtkTreeView*)(gtk_font_selection_get_face_list(gobj())));
}

const TreeView* FontSelection::get_face_list() const
{
  return const_cast<FontSelection*>(this)->get_face_list();
}

Entry* FontSelection::get_size_entry()
{
  return Glib::wrap((GtkEntry*)(gtk_font_selection_get_size_entry(gobj())));
}

const Entry* FontSelection::get_size_entry() const
{
  return const_cast<FontSelection*>(this)->get_size_entry();
}

TreeView* FontSelection::get_size_list()
{
  return Glib::wrap((GtkTreeView*)(gtk_font_selection_get_size_list(gobj())));
}

const TreeView* FontSelection::get_size_list() const
{
  return const_cast<FontSelection*>(this)->get_size_list();
}

Entry* FontSelection::get_preview_entry()
{
  return Glib::wrap((GtkEntry*)(gtk_font_selection_get_preview_entry(gobj())));
}

const Entry* FontSelection::get_preview_entry() const
{
  return Glib::wrap((GtkEntry*)(gtk_font_selection_get_preview_entry(const_cast<GtkFontSelection*>(gobj()))));
}

Glib::RefPtr<Pango::FontFamily> FontSelection::get_family()
{
  Glib::RefPtr<Pango::FontFamily> retvalue = Glib::wrap(gtk_font_selection_get_family(gobj()));
  if(retvalue)
    retvalue->reference(); //The function does not do a ref for us.
  return retvalue;
}

Glib::RefPtr<const Pango::FontFamily> FontSelection::get_family() const
{
  return const_cast<FontSelection*>(this)->get_family();
}

Glib::RefPtr<Pango::FontFace> FontSelection::get_face()
{
  Glib::RefPtr<Pango::FontFace> retvalue = Glib::wrap(gtk_font_selection_get_face(gobj()));
  if(retvalue)
    retvalue->reference(); //The function does not do a ref for us.
  return retvalue;
}

Glib::RefPtr<const Pango::FontFace> FontSelection::get_face() const
{
  return const_cast<FontSelection*>(this)->get_face();
}

int FontSelection::get_size() const
{
  return gtk_font_selection_get_size(const_cast<GtkFontSelection*>(gobj()));
}

Glib::ustring FontSelection::get_font_name() const
{
  return Glib::convert_return_gchar_ptr_to_ustring(gtk_font_selection_get_font_name(const_cast<GtkFontSelection*>(gobj())));
}

bool FontSelection::set_font_name(const Glib::ustring& fontname)
{
  return gtk_font_selection_set_font_name(gobj(), fontname.c_str());
}

Glib::ustring FontSelection::get_preview_text() const
{
  return Glib::convert_const_gchar_ptr_to_ustring(gtk_font_selection_get_preview_text(const_cast<GtkFontSelection*>(gobj())));
}

void FontSelection::set_preview_text(const Glib::ustring& text)
{
  gtk_font_selection_set_preview_text(gobj(), text.c_str());
}


Glib::PropertyProxy< Glib::ustring > FontSelection::property_font_name() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "font-name");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > FontSelection::property_font_name() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "font-name");
}

Glib::PropertyProxy< Glib::ustring > FontSelection::property_preview_text() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "preview-text");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > FontSelection::property_preview_text() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "preview-text");
}


} // namespace Gtk


namespace Glib
{

Gtk::FontSelectionDialog* wrap(GtkFontSelectionDialog* object, bool take_copy)
{
  return dynamic_cast<Gtk::FontSelectionDialog *> (Glib::wrap_auto ((GObject*)(object), take_copy));
}

} /* namespace Glib */

namespace Gtk
{


/* The *_Class implementation: */

const Glib::Class& FontSelectionDialog_Class::init()
{
  if(!gtype_) // create the GType if necessary
  {
    // Glib::Class has to know the class init function to clone custom types.
    class_init_func_ = &FontSelectionDialog_Class::class_init_function;

    // This is actually just optimized away, apparently with no harm.
    // Make sure that the parent type has been created.
    //CppClassParent::CppObjectType::get_type();

    // Create the wrapper type, with the same class/instance size as the base type.
    register_derived_type(gtk_font_selection_dialog_get_type());

    // Add derived versions of interfaces, if the C type implements any interfaces:

  }

  return *this;
}


void FontSelectionDialog_Class::class_init_function(void* g_class, void* class_data)
{
  const auto klass = static_cast<BaseClassType*>(g_class);
  CppClassParent::class_init_function(klass, class_data);


}


Glib::ObjectBase* FontSelectionDialog_Class::wrap_new(GObject* o)
{
  return new FontSelectionDialog((GtkFontSelectionDialog*)(o)); //top-level windows can not be manage()ed.

}


/* The implementation: */

FontSelectionDialog::FontSelectionDialog(const Glib::ConstructParams& construct_params)
:
  Gtk::Dialog(construct_params)
{
  }

FontSelectionDialog::FontSelectionDialog(GtkFontSelectionDialog* castitem)
:
  Gtk::Dialog((GtkDialog*)(castitem))
{
  }


FontSelectionDialog::FontSelectionDialog(FontSelectionDialog&& src) noexcept
: Gtk::Dialog(std::move(src))
{}

FontSelectionDialog& FontSelectionDialog::operator=(FontSelectionDialog&& src) noexcept
{
  Gtk::Dialog::operator=(std::move(src));
  return *this;
}

FontSelectionDialog::~FontSelectionDialog() noexcept
{
  destroy_();
}

FontSelectionDialog::CppClassType FontSelectionDialog::fontselectiondialog_class_; // initialize static member

GType FontSelectionDialog::get_type()
{
  return fontselectiondialog_class_.init().get_type();
}


GType FontSelectionDialog::get_base_type()
{
  return gtk_font_selection_dialog_get_type();
}

FontSelectionDialog::FontSelectionDialog()
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  Gtk::Dialog(Glib::ConstructParams(fontselectiondialog_class_.init()))
{
  

}

FontSelectionDialog::FontSelectionDialog(const Glib::ustring& title)
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  Gtk::Dialog(Glib::ConstructParams(fontselectiondialog_class_.init(), "title", title.c_str(), nullptr))
{
  

}

bool FontSelectionDialog::set_font_name(const Glib::ustring& fontname)
{
  return gtk_font_selection_dialog_set_font_name(gobj(), fontname.c_str());
}

Glib::ustring FontSelectionDialog::get_font_name() const
{
  return Glib::convert_return_gchar_ptr_to_ustring(gtk_font_selection_dialog_get_font_name(const_cast<GtkFontSelectionDialog*>(gobj())));
}

Glib::ustring FontSelectionDialog::get_preview_text() const
{
  return Glib::convert_const_gchar_ptr_to_ustring(gtk_font_selection_dialog_get_preview_text(const_cast<GtkFontSelectionDialog*>(gobj())));
}

void FontSelectionDialog::set_preview_text(const Glib::ustring& text)
{
  gtk_font_selection_dialog_set_preview_text(gobj(), text.c_str());
}

Button* FontSelectionDialog::get_ok_button()
{
  return Glib::wrap((GtkButton*)(gtk_font_selection_dialog_get_ok_button(gobj())));
}

const Button* FontSelectionDialog::get_ok_button() const
{
  return const_cast<FontSelectionDialog*>(this)->get_ok_button();
}

Button* FontSelectionDialog::get_cancel_button()
{
  return Glib::wrap((GtkButton*)(gtk_font_selection_dialog_get_cancel_button(gobj())));
}

const Button* FontSelectionDialog::get_cancel_button() const
{
  return const_cast<FontSelectionDialog*>(this)->get_cancel_button();
}


} // namespace Gtk

#endif // GTKMM_DISABLE_DEPRECATED


