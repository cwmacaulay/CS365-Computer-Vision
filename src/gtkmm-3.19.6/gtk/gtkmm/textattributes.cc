// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!


#include <glibmm.h>

#include <gtkmm/textattributes.h>
#include <gtkmm/private/textattributes_p.h>


/*
 * Copyright 1998-2002 The gtkmm Development Team
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

#include <gtk/gtk.h>


namespace
{
} // anonymous namespace


namespace Glib
{

Gtk::TextAttributes wrap(GtkTextAttributes* object, bool take_copy)
{
  return Gtk::TextAttributes(object, take_copy);
}

} // namespace Glib


namespace Gtk
{


// static
GType TextAttributes::get_type()
{
  return gtk_text_attributes_get_type();
}

TextAttributes::TextAttributes()
:
  gobject_ (gtk_text_attributes_new())
{}

TextAttributes::TextAttributes(const TextAttributes& other)
:
  gobject_ ((other.gobject_) ? gtk_text_attributes_copy(other.gobject_) : nullptr)
{}

TextAttributes::TextAttributes(TextAttributes&& other) noexcept
:
  gobject_(other.gobject_)
{
  other.gobject_ = nullptr;
}

TextAttributes& TextAttributes::operator=(TextAttributes&& other) noexcept
{
  TextAttributes temp (other);
  swap(temp);
  return *this;
}

TextAttributes::TextAttributes(GtkTextAttributes* gobject, bool make_a_copy)
:
  // For BoxedType wrappers, make_a_copy is true by default.  The static
  // BoxedType wrappers must always take a copy, thus make_a_copy = true
  // ensures identical behaviour if the default argument is used.
  gobject_ ((make_a_copy && gobject) ? gtk_text_attributes_copy(gobject) : gobject)
{}

TextAttributes& TextAttributes::operator=(const TextAttributes& other)
{
  TextAttributes temp (other);
  swap(temp);
  return *this;
}

TextAttributes::~TextAttributes() noexcept
{
  if(gobject_)
    gtk_text_attributes_unref(gobject_);
}

void TextAttributes::swap(TextAttributes& other) noexcept
{
  GtkTextAttributes *const temp = gobject_;
  gobject_ = other.gobject_;
  other.gobject_ = temp;
}

GtkTextAttributes* TextAttributes::gobj_copy() const
{
  return gtk_text_attributes_copy(gobject_);
}


} // namespace Gtk


