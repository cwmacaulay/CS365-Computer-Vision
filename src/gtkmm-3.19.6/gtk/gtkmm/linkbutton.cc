// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!


#include <glibmm.h>

#include <gtkmm/linkbutton.h>
#include <gtkmm/private/linkbutton_p.h>


/*
 * Copyright 2006 The gtkmm Development Team
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


static gboolean LinkButton_signal_activate_link_callback(GtkLinkButton* self, void* data)
{
  using namespace Gtk;
  typedef sigc::slot< bool > SlotType;

  auto obj = dynamic_cast<LinkButton*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        return static_cast<int>((*static_cast<SlotType*>(slot))());
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }

  typedef gboolean RType;
  return RType();
}

static gboolean LinkButton_signal_activate_link_notify_callback(GtkLinkButton* self,  void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void > SlotType;

  auto obj = dynamic_cast<LinkButton*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))();
    }
    catch(...)
    {
      Glib::exception_handlers_invoke();
    }
  }

  typedef gboolean RType;
  return RType();
}

static const Glib::SignalProxyInfo LinkButton_signal_activate_link_info =
{
  "activate-link",
  (GCallback) &LinkButton_signal_activate_link_callback,
  (GCallback) &LinkButton_signal_activate_link_notify_callback
};


} // anonymous namespace


namespace Glib
{

Gtk::LinkButton* wrap(GtkLinkButton* object, bool take_copy)
{
  return dynamic_cast<Gtk::LinkButton *> (Glib::wrap_auto ((GObject*)(object), take_copy));
}

} /* namespace Glib */

namespace Gtk
{


/* The *_Class implementation: */

const Glib::Class& LinkButton_Class::init()
{
  if(!gtype_) // create the GType if necessary
  {
    // Glib::Class has to know the class init function to clone custom types.
    class_init_func_ = &LinkButton_Class::class_init_function;

    // This is actually just optimized away, apparently with no harm.
    // Make sure that the parent type has been created.
    //CppClassParent::CppObjectType::get_type();

    // Create the wrapper type, with the same class/instance size as the base type.
    register_derived_type(gtk_link_button_get_type());

    // Add derived versions of interfaces, if the C type implements any interfaces:

  }

  return *this;
}


void LinkButton_Class::class_init_function(void* g_class, void* class_data)
{
  const auto klass = static_cast<BaseClassType*>(g_class);
  CppClassParent::class_init_function(klass, class_data);


  klass->activate_link = &activate_link_callback;
}


gboolean LinkButton_Class::activate_link_callback(GtkLinkButton* self)
{
  const auto obj_base = static_cast<Glib::ObjectBase*>(
      Glib::ObjectBase::_get_current_wrapper((GObject*)self));

  // Non-gtkmmproc-generated custom classes implicitly call the default
  // Glib::ObjectBase constructor, which sets is_derived_. But gtkmmproc-
  // generated classes can use this optimisation, which avoids the unnecessary
  // parameter conversions if there is no possibility of the virtual function
  // being overridden:
  if(obj_base && obj_base->is_derived_())
  {
    const auto obj = dynamic_cast<CppObjectType* const>(obj_base);
    if(obj) // This can be NULL during destruction.
    {
      try // Trap C++ exceptions which would normally be lost because this is a C callback.
      {
        // Call the virtual member method, which derived classes might override.
        return static_cast<int>(obj->on_activate_link());
      }
      catch(...)
      {
        Glib::exception_handlers_invoke();
      }
    }
  }

  const auto base = static_cast<BaseClassType*>(
        g_type_class_peek_parent(G_OBJECT_GET_CLASS(self)) // Get the parent class of the object class (The original underlying C class).
    );

  // Call the original underlying C function:
  if(base && base->activate_link)
    return (*base->activate_link)(self);

  typedef gboolean RType;
  return RType();
}


Glib::ObjectBase* LinkButton_Class::wrap_new(GObject* o)
{
  return manage(new LinkButton((GtkLinkButton*)(o)));

}


/* The implementation: */

LinkButton::LinkButton(const Glib::ConstructParams& construct_params)
:
  Gtk::Button(construct_params)
{
  }

LinkButton::LinkButton(GtkLinkButton* castitem)
:
  Gtk::Button((GtkButton*)(castitem))
{
  }


LinkButton::LinkButton(LinkButton&& src) noexcept
: Gtk::Button(std::move(src))
{}

LinkButton& LinkButton::operator=(LinkButton&& src) noexcept
{
  Gtk::Button::operator=(std::move(src));
  return *this;
}

LinkButton::~LinkButton() noexcept
{
  destroy_();
}

LinkButton::CppClassType LinkButton::linkbutton_class_; // initialize static member

GType LinkButton::get_type()
{
  return linkbutton_class_.init().get_type();
}


GType LinkButton::get_base_type()
{
  return gtk_link_button_get_type();
}


LinkButton::LinkButton()
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  Gtk::Button(Glib::ConstructParams(linkbutton_class_.init()))
{
  

}

LinkButton::LinkButton(const Glib::ustring& uri, const Glib::ustring& label)
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  Gtk::Button(Glib::ConstructParams(linkbutton_class_.init(), "uri", uri.c_str(), "label", label.c_str(), nullptr))
{
  

}

Glib::ustring LinkButton::get_uri() const
{
  return Glib::convert_const_gchar_ptr_to_ustring(gtk_link_button_get_uri(const_cast<GtkLinkButton*>(gobj())));
}

void LinkButton::set_uri(const Glib::ustring& uri)
{
  gtk_link_button_set_uri(gobj(), uri.c_str());
}

bool LinkButton::get_visited() const
{
  return gtk_link_button_get_visited(const_cast<GtkLinkButton*>(gobj()));
}

void LinkButton::set_visited(bool visited)
{
  gtk_link_button_set_visited(gobj(), static_cast<int>(visited));
}


Glib::SignalProxy0< bool > LinkButton::signal_activate_link()
{
  return Glib::SignalProxy0< bool >(this, &LinkButton_signal_activate_link_info);
}


Glib::PropertyProxy< Glib::ustring > LinkButton::property_uri() 
{
  return Glib::PropertyProxy< Glib::ustring >(this, "uri");
}

Glib::PropertyProxy_ReadOnly< Glib::ustring > LinkButton::property_uri() const
{
  return Glib::PropertyProxy_ReadOnly< Glib::ustring >(this, "uri");
}

Glib::PropertyProxy< bool > LinkButton::property_visited() 
{
  return Glib::PropertyProxy< bool >(this, "visited");
}

Glib::PropertyProxy_ReadOnly< bool > LinkButton::property_visited() const
{
  return Glib::PropertyProxy_ReadOnly< bool >(this, "visited");
}


bool Gtk::LinkButton::on_activate_link()
{
  const auto base = static_cast<BaseClassType*>(
      g_type_class_peek_parent(G_OBJECT_GET_CLASS(gobject_)) // Get the parent class of the object class (The original underlying C class).
  );

  if(base && base->activate_link)
    return (*base->activate_link)(gobj());

  typedef bool RType;
  return RType();
}


} // namespace Gtk


