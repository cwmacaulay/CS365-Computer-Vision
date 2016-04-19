// Generated by gmmproc 2.47.4 -- DO NOT MODIFY!


#include <glibmm.h>

#include <gtkmm/scalebutton.h>
#include <gtkmm/private/scalebutton_p.h>


/*
 * Copyright 2007 The gtkmm Development Team
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

#include <glibmm/vectorutils.h>

#include <gtk/gtk.h>

namespace Gtk
{

ScaleButton::ScaleButton(IconSize size, double min, double max, double step, const std::vector<Glib::ustring>& icons)
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  Gtk::Button(Glib::ConstructParams(scalebutton_class_.init(), "size",static_cast<GtkIconSize>(int(size)),"icons",Glib::ArrayHandler<Glib::ustring>::vector_to_array(icons).data(),nullptr, nullptr))
{
  auto adjustment = Adjustment::create(min, min, max, step, 10 * step, 0);
  set_adjustment(adjustment);
}

ScaleButton::ScaleButton(IconSize size, double min, double max, double step)
:
  // Mark this class as non-derived to allow C++ vfuncs to be skipped.
  Glib::ObjectBase(nullptr),
  Gtk::Button(Glib::ConstructParams(scalebutton_class_.init(), "size",static_cast<GtkIconSize>(int(size)), nullptr))
{
  auto adjustment = Adjustment::create(min, min, max, step, 10 * step, 0);
  set_adjustment(adjustment);
}

} // namespace Gtk


namespace
{


static void ScaleButton_signal_value_changed_callback(GtkScaleButton* self, gdouble p0,void* data)
{
  using namespace Gtk;
  typedef sigc::slot< void,double > SlotType;

  auto obj = dynamic_cast<ScaleButton*>(Glib::ObjectBase::_get_current_wrapper((GObject*) self));
  // Do not try to call a signal on a disassociated wrapper.
  if(obj)
  {
    try
    {
      if(const auto slot = Glib::SignalProxyNormal::data_to_slot(data))
        (*static_cast<SlotType*>(slot))(p0
);
    }
    catch(...)
    {
       Glib::exception_handlers_invoke();
    }
  }
}

static const Glib::SignalProxyInfo ScaleButton_signal_value_changed_info =
{
  "value_changed",
  (GCallback) &ScaleButton_signal_value_changed_callback,
  (GCallback) &ScaleButton_signal_value_changed_callback
};


} // anonymous namespace


namespace Glib
{

Gtk::ScaleButton* wrap(GtkScaleButton* object, bool take_copy)
{
  return dynamic_cast<Gtk::ScaleButton *> (Glib::wrap_auto ((GObject*)(object), take_copy));
}

} /* namespace Glib */

namespace Gtk
{


/* The *_Class implementation: */

const Glib::Class& ScaleButton_Class::init()
{
  if(!gtype_) // create the GType if necessary
  {
    // Glib::Class has to know the class init function to clone custom types.
    class_init_func_ = &ScaleButton_Class::class_init_function;

    // This is actually just optimized away, apparently with no harm.
    // Make sure that the parent type has been created.
    //CppClassParent::CppObjectType::get_type();

    // Create the wrapper type, with the same class/instance size as the base type.
    register_derived_type(gtk_scale_button_get_type());

    // Add derived versions of interfaces, if the C type implements any interfaces:
  Orientable::add_interface(get_type());

  }

  return *this;
}


void ScaleButton_Class::class_init_function(void* g_class, void* class_data)
{
  const auto klass = static_cast<BaseClassType*>(g_class);
  CppClassParent::class_init_function(klass, class_data);


  klass->value_changed = &value_changed_callback;
}


void ScaleButton_Class::value_changed_callback(GtkScaleButton* self, gdouble p0)
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
        obj->on_value_changed(p0
);
        return;
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
  if(base && base->value_changed)
    (*base->value_changed)(self, p0);
}


Glib::ObjectBase* ScaleButton_Class::wrap_new(GObject* o)
{
  return manage(new ScaleButton((GtkScaleButton*)(o)));

}


/* The implementation: */

ScaleButton::ScaleButton(const Glib::ConstructParams& construct_params)
:
  Gtk::Button(construct_params)
{
  }

ScaleButton::ScaleButton(GtkScaleButton* castitem)
:
  Gtk::Button((GtkButton*)(castitem))
{
  }


ScaleButton::ScaleButton(ScaleButton&& src) noexcept
: Gtk::Button(std::move(src))
  , Orientable(std::move(src))
{}

ScaleButton& ScaleButton::operator=(ScaleButton&& src) noexcept
{
  Gtk::Button::operator=(std::move(src));
  Orientable::operator=(std::move(src));
  return *this;
}

ScaleButton::~ScaleButton() noexcept
{
  destroy_();
}

ScaleButton::CppClassType ScaleButton::scalebutton_class_; // initialize static member

GType ScaleButton::get_type()
{
  return scalebutton_class_.init().get_type();
}


GType ScaleButton::get_base_type()
{
  return gtk_scale_button_get_type();
}


void ScaleButton::set_icons(const std::vector<Glib::ustring>& icons)
{
  gtk_scale_button_set_icons(gobj(), Glib::ArrayHandler<Glib::ustring>::vector_to_array(icons).data ());
}

double ScaleButton::get_value() const
{
  return gtk_scale_button_get_value(const_cast<GtkScaleButton*>(gobj()));
}

void ScaleButton::set_value(double value)
{
  gtk_scale_button_set_value(gobj(), value);
}

Glib::RefPtr<Adjustment> ScaleButton::get_adjustment()
{
  Glib::RefPtr<Adjustment> retvalue = Glib::wrap(gtk_scale_button_get_adjustment(gobj()));
  if(retvalue)
    retvalue->reference(); //The function does not do a ref for us.
  return retvalue;
}

Glib::RefPtr<const Adjustment> ScaleButton::get_adjustment() const
{
  return const_cast<ScaleButton*>(this)->get_adjustment();
}

void ScaleButton::set_adjustment(const Glib::RefPtr<Adjustment>& adjustment)
{
  gtk_scale_button_set_adjustment(gobj(), Glib::unwrap(adjustment));
}

Widget* ScaleButton::get_plus_button()
{
  return Glib::wrap(gtk_scale_button_get_plus_button(gobj()));
}

const Widget* ScaleButton::get_plus_button() const
{
  return const_cast<ScaleButton*>(this)->get_plus_button();
}

Widget* ScaleButton::get_minus_button()
{
  return Glib::wrap(gtk_scale_button_get_minus_button(gobj()));
}

const Widget* ScaleButton::get_minus_button() const
{
  return const_cast<ScaleButton*>(this)->get_minus_button();
}

Gtk::Widget* ScaleButton::get_popup()
{
  return Glib::wrap(gtk_scale_button_get_popup(gobj()));
}

const Gtk::Widget* ScaleButton::get_popup() const
{
  return Glib::wrap(gtk_scale_button_get_popup(const_cast<GtkScaleButton*>(gobj())));
}


Glib::SignalProxy1< void,double > ScaleButton::signal_value_changed()
{
  return Glib::SignalProxy1< void,double >(this, &ScaleButton_signal_value_changed_info);
}


Glib::PropertyProxy< double > ScaleButton::property_value() 
{
  return Glib::PropertyProxy< double >(this, "value");
}

Glib::PropertyProxy_ReadOnly< double > ScaleButton::property_value() const
{
  return Glib::PropertyProxy_ReadOnly< double >(this, "value");
}

Glib::PropertyProxy< IconSize > ScaleButton::property_size() 
{
  return Glib::PropertyProxy< IconSize >(this, "size");
}

Glib::PropertyProxy_ReadOnly< IconSize > ScaleButton::property_size() const
{
  return Glib::PropertyProxy_ReadOnly< IconSize >(this, "size");
}

Glib::PropertyProxy< Adjustment* > ScaleButton::property_adjustment() 
{
  return Glib::PropertyProxy< Adjustment* >(this, "adjustment");
}

Glib::PropertyProxy_ReadOnly< Adjustment* > ScaleButton::property_adjustment() const
{
  return Glib::PropertyProxy_ReadOnly< Adjustment* >(this, "adjustment");
}

Glib::PropertyProxy< std::vector<Glib::ustring> > ScaleButton::property_icons() 
{
  return Glib::PropertyProxy< std::vector<Glib::ustring> >(this, "icons");
}

Glib::PropertyProxy_ReadOnly< std::vector<Glib::ustring> > ScaleButton::property_icons() const
{
  return Glib::PropertyProxy_ReadOnly< std::vector<Glib::ustring> >(this, "icons");
}


void Gtk::ScaleButton::on_value_changed(double value)
{
  const auto base = static_cast<BaseClassType*>(
      g_type_class_peek_parent(G_OBJECT_GET_CLASS(gobject_)) // Get the parent class of the object class (The original underlying C class).
  );

  if(base && base->value_changed)
    (*base->value_changed)(gobj(),value);
}


} // namespace Gtk


